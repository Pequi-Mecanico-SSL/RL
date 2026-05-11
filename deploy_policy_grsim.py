#!/usr/bin/env python3
"""
Deploy de Policy RL no grSim via UDP com observações reais do SSL-Vision

Este script carrega um checkpoint RLlib e executa a política
no simulador grSim enviando comandos UDP na porta 20011,
recebendo dados de vision na porta UDP padrao do grSim (10020 / multicast 224.5.23.2)

Uso:
    python deploy_policy_grsim.py

Variáveis de ambiente:
    CHECKPOINT_PATH: Caminho para o checkpoint RLlib
    TEAM: Time a controlar ('blue' ou 'yellow')
    GRSIM_HOST: Host do grSim (default: grsim)
    GRSIM_PORT: Porta de comandos UDP (default: 20011)
    VISION_PORT: Porta UDP SSL-Vision do grSim (default: 10020; conferir Communication no grSim)
    FPS: Frames por segundo (default: 30)
    DEVICE: Dispositivo PyTorch ('cpu' ou 'cuda')
"""

import os
import sys
import pickle
import time
import logging
import socket
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# Adicionar path para arquivos protobuf
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar arquivos protobuf do grSim
try:
    import grSim_Packet_pb2 as grSim_Packet
    import grSim_Commands_pb2 as grSim_Commands
    GRSIM_PROTO_AVAILABLE = True
except ImportError as e:
    GRSIM_PROTO_AVAILABLE = False
    print(f"Warning: grSim protobuf files not available: {e}")

# Importar arquivos protobuf do SSL-Vision
try:
    import ssl_vision_wrapper_pb2 as ssl_vision_wrapper
    import ssl_vision_detection_pb2 as ssl_vision_detection
    import ssl_vision_geometry_pb2 as ssl_vision_geometry
    SSL_VISION_PROTO_AVAILABLE = True
except ImportError as e:
    SSL_VISION_PROTO_AVAILABLE = False
    print(f"Warning: SSL-Vision protobuf files not available: {e}")

# Importar funções de observações
sys.path.insert(0, '/app/rSoccer')
try:
    from observations import (
        positions_observations,
        oritations_observations,
        distances_observations,
        angles_observations,
        timesteps_observations,
        actions_observations
    )
    from rsoccer_gym.Utils.Utils import Geometry2D
    OBSERVATIONS_AVAILABLE = True
    print("Observations module loaded successfully")
except ImportError as e:
    OBSERVATIONS_AVAILABLE = False
    print(f"Warning: observations module not available: {e}")

from scripts.model.model_inferece import InferenceModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/policy.log') if os.path.exists('/app/logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger('deploy_policy_grsim')


@dataclass
class DeployConfig:
    """Configuracao para deploy."""
    checkpoint_path: str
    team: str = "blue"
    grsim_host: str = "grsim"
    grsim_port: int = 20011
    vision_port: int = 10020
    vision_address: str = "224.5.23.2"
    fps: int = 30
    device: str = "cpu"
    n_robots_blue: int = 3
    n_robots_yellow: int = 3
    field_type: int = 1
    obs_size: int = 77
    n_stack: int = 8
    action_size: int = 4


class InferenceBetaDist:
    """Distribuicao Beta para amostragem de acoes."""

    def __init__(self, inputs: torch.Tensor, signal: List[float] = None):
        self.inputs = inputs
        self.inputs = torch.clamp(self.inputs, -20, 20)
        self.inputs = torch.log(torch.exp(self.inputs) + 1.0) + 1.0

        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        self.signal = torch.tensor(signal or [1, 1, 1, 1], dtype=torch.float32)

    def sample(self) -> torch.Tensor:
        sample = self.dist.rsample()
        sample = sample * 2.0 - 1.0
        return self.signal.to(sample.device) * sample

    def mean(self) -> torch.Tensor:
        mean = self.dist.mean
        mean = mean * 2.0 - 1.0
        return self.signal.to(mean.device) * mean


class GrSimVisionController:
    """Controlador que recebe vision SSL e envia comandos UDP para o grSim."""

    def __init__(self, config: DeployConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        logger.info(f"Inicializando GrSimVisionController:")
        logger.info(f"  Checkpoint: {config.checkpoint_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Team: {config.team}")
        logger.info(f"  grSim: {config.grsim_host}:{config.grsim_port}")
        logger.info(f"  SSL-Vision proto: {SSL_VISION_PROTO_AVAILABLE}")
        logger.info(f"  Observations module: {OBSERVATIONS_AVAILABLE}")

        # Carregar modelo
        self.model = self._load_model()
        self.model.eval()

        # Inicializar buffers de observações (stack de 8 frames)
        self.stacked_obs = self._init_stacked_obs()
        self.last_actions = self._init_actions()

        # Criar socket UDP para enviar comandos
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Criar socket UDP para receber vision (multicast)
        self.vision_socket = None
        self.vision_data = None
        self.field_info = None
        self._vision_first_packet_logged = False
        self._init_vision_socket()

        # Contadores
        self.step_count = 0
        self.inference_times = []
        self.vision_packets_received = 0

    def _init_vision_socket(self):
        """Inicializa socket UDP para receber SSL-Vision (multicast).

        Usa SO_REUSEADDR e, no Linux, SO_REUSEPORT para dois containers (blue/yellow)
        no mesmo host escutarem a mesma porta multicast.

        Variaveis opcionais:
            VISION_BIND_ADDR: IP para bind (default todas as interfaces: '')
        """
        if not SSL_VISION_PROTO_AVAILABLE:
            logger.warning("SSL-Vision protobuf not available")
            return

        bind_addr = os.environ.get("VISION_BIND_ADDR", "").strip()

        try:
            self.vision_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.vision_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Linux: permite dois processos no mesmo host (rl-policy + rl-policy-yellow)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    self.vision_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError as e:
                    logger.warning(f"SO_REUSEPORT nao aplicado: {e}")

            self.vision_socket.bind((bind_addr, self.config.vision_port))

            # Multicast: grupo + interface (0.0.0.0 = kernel escolhe)
            mreq = struct.pack(
                "4sl",
                socket.inet_aton(self.config.vision_address),
                socket.INADDR_ANY,
            )
            self.vision_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            self.vision_socket.setblocking(False)
            logger.info(
                f"Vision socket: bind={bind_addr or '0.0.0.0'}:{self.config.vision_port}, "
                f"multicast={self.config.vision_address}"
            )
        except Exception as e:
            logger.warning(f"Could not initialize vision socket: {e}")
            self.vision_socket = None

    def _load_model(self) -> InferenceModel:
        """Carrega o modelo do checkpoint RLlib."""
        model = InferenceModel(
            input_size=self.config.obs_size * self.config.n_stack,
            output_size=2 * self.config.action_size
        )

        policy_file = Path(self.config.checkpoint_path) / "policies/policy_blue/policy_state.pkl"

        if not policy_file.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado: {policy_file}")

        logger.info(f"Carregando checkpoint: {policy_file}")

        with open(policy_file, "rb") as f:
            policy_state = pickle.load(f)

        weights_dict = {}
        for layer_name, weights in policy_state["weights"].items():
            split = layer_name.split(".")
            if "_logits" == split[0] or "_value_branch" == split[0]:
                new_layer_name = split[0] + "." + split[-1]
            else:
                if len(split) >= 3:
                    layer_idx = int(split[1])
                    new_layer_name = split[0] + "." + str(layer_idx * 2) + "." + split[-1]
                else:
                    new_layer_name = layer_name
            weights_dict[new_layer_name] = torch.tensor(weights)

        model.load_state_dict(weights_dict, strict=False)
        model.to(self.device)

        logger.info("Modelo carregado com sucesso")
        return model

    def _init_stacked_obs(self) -> Dict[str, np.ndarray]:
        """Inicializa o buffer de observacoes."""
        return {
            **{f"blue_{i}": np.zeros(self.config.obs_size * self.config.n_stack, dtype=np.float32)
               for i in range(self.config.n_robots_blue)},
            **{f"yellow_{i}": np.zeros(self.config.obs_size * self.config.n_stack, dtype=np.float32)
               for i in range(self.config.n_robots_yellow)}
        }

    def _init_actions(self) -> Dict[str, np.ndarray]:
        """Inicializa as acoes."""
        return {
            **{f"blue_{i}": np.zeros(self.config.action_size, dtype=np.float32)
               for i in range(self.config.n_robots_blue)},
            **{f"yellow_{i}": np.zeros(self.config.action_size, dtype=np.float32)
               for i in range(self.config.n_robots_yellow)}
        }

    def _parse_vision_datagram(self, data: bytes) -> Optional[Dict]:
        """Parse um buffer UDP SSL-Vision num frame dict ou None."""
        wrapper = ssl_vision_wrapper.SSL_WrapperPacket()
        wrapper.ParseFromString(data)
        self.vision_packets_received += 1
        if not self._vision_first_packet_logged:
            self._vision_first_packet_logged = True
            logger.info(f"Primeiro pacote SSL-Vision UDP recebido ({len(data)} bytes).")

        if not (
            wrapper.detection.robots_blue
            or wrapper.detection.robots_yellow
            or wrapper.detection.balls
        ):
            return None

        detection = wrapper.detection

        frame = {
            "robots_blue": {},
            "robots_yellow": {},
            "ball": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        for robot in detection.robots_blue:
            frame["robots_blue"][f"robot_{robot.robot_id}"] = {
                "x": robot.x / 1000.0,
                "y": robot.y / 1000.0,
                "theta": np.rad2deg(robot.orientation),
            }

        for robot in detection.robots_yellow:
            frame["robots_yellow"][f"robot_{robot.robot_id}"] = {
                "x": robot.x / 1000.0,
                "y": robot.y / 1000.0,
                "theta": np.rad2deg(robot.orientation),
            }

        if detection.balls:
            ball = detection.balls[0]
            frame["ball"] = {
                "x": ball.x / 1000.0,
                "y": ball.y / 1000.0,
                "z": ball.z / 1000.0 if ball.z else 0.0,
            }

        # SSL-Vision pode mandar pacotes contendo so blue, so yellow ou so ball.
        # Sempre que parsear ao menos UMA deteccao, mantemos o frame mas mantendo
        # os ids ausentes vazios (caller faz fallback para coordenadas zero).
        if wrapper.geometry.field:
            field = wrapper.geometry.field
            self.field_info = {
                "length": field.field_length / 1000.0,
                "width": field.field_width / 1000.0,
                "goal_width": field.goal_width / 1000.0 if field.goal_width else 1.0,
            }

        return frame

    def _receive_vision_data(self) -> Optional[Dict]:
        """Recebe e parseia dados do SSL-Vision (esvazia fila UDP; usa ultimo frame valido)."""
        if self.vision_socket is None or not SSL_VISION_PROTO_AVAILABLE:
            return None

        last_frame: Optional[Dict] = None
        n_drained = 0
        n_parsed_ok = 0
        n_with_det = 0
        try:
            while True:
                try:
                    data, _addr = self.vision_socket.recvfrom(65535)
                except BlockingIOError:
                    break
                n_drained += 1
                try:
                    frame = self._parse_vision_datagram(data)
                    n_parsed_ok += 1
                    if frame is not None:
                        n_with_det += 1
                        last_frame = frame
                        if not self._vision_first_packet_logged:
                            self._vision_first_packet_logged = True
                            logger.info(
                                "Primeiro pacote SSL-Vision parseado com deteccao (robos/bola)."
                            )
                except Exception as e:
                    logger.debug(f"Vision parse error: {e}")

            if self.step_count % 100 == 0 and n_drained > 0:
                logger.debug(
                    f"vision drain: drained={n_drained} parsed_ok={n_parsed_ok} with_det={n_with_det}"
                )
            return last_frame

        except Exception as e:
            logger.debug(f"Vision receive error: {e}")
            return None

    def _build_observations_from_frame(self, frame: Dict) -> Dict[str, np.ndarray]:
        """Constroi observacoes a partir do frame do SSL-Vision.

        IMPORTANTE: as funcoes em observations.py sao decoradas por
        decorator_observations (rsoccer_gym.Utils.Utils), que tem signature
        externa (n_blue, n_yellow, raw_observations, field_info, kwargs)
        e retorna {f"blue_<i>": obs_array, ..., f"yellow_<j>": obs_array}
        ja com a inversion trick em X para o time amarelo.
        """
        if frame is None or not OBSERVATIONS_AVAILABLE:
            # Fallback: manter observacoes anteriores
            return self.stacked_obs

        # Usar valores padrao de campo se geometria nao chegou ou veio zerada
        if (
            self.field_info is None
            or not self.field_info.get("length")
            or not self.field_info.get("width")
        ):
            self.field_info = {"length": 12.0, "width": 9.0, "goal_width": 1.0}

        n_blue = self.config.n_robots_blue
        n_yellow = self.config.n_robots_yellow
        team = self.config.team

        # raw_observations no formato esperado pelo decorator: dict por nome
        raw = {}
        for i in range(n_blue):
            raw[f"blue_{i}"] = frame["robots_blue"].get(
                f"robot_{i}", {"x": 0.0, "y": 0.0, "theta": 0.0}
            )
        for j in range(n_yellow):
            raw[f"yellow_{j}"] = frame["robots_yellow"].get(
                f"robot_{j}", {"x": 0.0, "y": 0.0, "theta": 0.0}
            )
        raw["ball"] = frame["ball"]

        # kwargs comum a todas as funcoes (cada feature usa o subconjunto que precisa)
        full_kwargs = {
            "field_info": self.field_info,
            "steps": self.step_count,
            "max_ep_length": 1800,
            "last_actions": self.last_actions,
        }

        # Lista (funcao, kwargs_required) espelhando OBSERVATIONS de observations.py
        observation_funcs = [
            (positions_observations, ["field_info"]),
            (oritations_observations, []),
            (distances_observations, ["field_info"]),
            (angles_observations, ["field_info"]),
            (timesteps_observations, ["max_ep_length", "steps"]),
            (actions_observations, ["last_actions"]),
        ]

        # Cada funcao retorna {robot_name: feature_array} para todos os robos.
        per_feature_dicts: List[Dict[str, np.ndarray]] = []
        try:
            for func, required in observation_funcs:
                func_kwargs = {k: full_kwargs[k] for k in required if k in full_kwargs}
                per_feature_dicts.append(
                    func(n_blue, n_yellow, raw, self.field_info, func_kwargs)
                )
        except Exception as e:
            logger.warning(f"Falha ao computar features de observacao: {e}")
            return self.stacked_obs

        # Concatena features por robo do time controlado e atualiza o stack
        observations = {}
        for robot_id in range(n_blue if team == "blue" else n_yellow):
            robot_key = f"{team}_{robot_id}"
            try:
                new_obs = np.concatenate(
                    [d[robot_key].astype(np.float32) for d in per_feature_dicts]
                )
            except KeyError as e:
                logger.warning(f"Robo {robot_key} ausente nas features: {e}")
                observations[robot_key] = self.stacked_obs[robot_key]
                continue

            if len(new_obs) < self.config.obs_size:
                new_obs = np.pad(new_obs, (0, self.config.obs_size - len(new_obs)))
            elif len(new_obs) > self.config.obs_size:
                new_obs = new_obs[: self.config.obs_size]

            old_stacked = self.stacked_obs[robot_key]
            new_stacked = np.roll(old_stacked, -self.config.obs_size)
            new_stacked[-self.config.obs_size:] = new_obs
            observations[robot_key] = new_stacked

        return observations

    def _compute_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """Computa acoes usando o modelo."""
        start_time = time.time()

        model_inputs = []
        robot_names = []

        team = self.config.team
        for robot_name in sorted(observations.keys()):
            if team in robot_name:
                model_inputs.append(observations[robot_name])
                robot_names.append(robot_name)

        if not model_inputs:
            return {}

        model_input = torch.tensor(
            np.array(model_inputs, dtype=np.float32)
        ).to(self.device)

        with torch.no_grad():
            model_output, value = self.model(model_input)

        signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
        distribution = InferenceBetaDist(model_output, signal=signal)

        actions_tensor = distribution.mean()
        actions = actions_tensor.detach().cpu().numpy()

        action_dict = {}
        for i, robot_name in enumerate(robot_names):
            action_dict[robot_name] = actions[i].tolist()

        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)

        if len(self.inference_times) >= 100:
            avg_time = np.mean(self.inference_times[-100:])
            logger.debug(f"Tempo medio de inferencia: {avg_time:.2f}ms")

        return action_dict

    def _send_commands_udp(self, actions: Dict[str, List[float]]):
        """Envia comandos UDP para o grSim."""
        if not GRSIM_PROTO_AVAILABLE:
            logger.warning("Protobuf not available, cannot send commands")
            return

        try:
            packet = grSim_Packet.grSim_Packet()
            commands = packet.commands

            commands.timestamp = time.time()
            commands.isteamyellow = (self.config.team == "yellow")

            max_linear = 5.0
            max_angular = 20.0
            kick_speed = 3.0

            for robot_name, action in actions.items():
                color, idx_str = robot_name.split("_")
                idx = int(idx_str)

                if color != self.config.team:
                    continue

                robot_cmd = commands.robot_commands.add()
                robot_cmd.id = idx
                robot_cmd.wheelsspeed = False
                robot_cmd.veltangent = action[0] * max_linear
                robot_cmd.velnormal = action[1] * max_linear
                robot_cmd.velangular = action[2] * max_angular
                robot_cmd.kickspeedx = kick_speed if action[3] > 0.5 else 0.0
                robot_cmd.kickspeedz = 0.0
                robot_cmd.spinner = False

            data = packet.SerializeToString()
            self.command_socket.sendto(data, (self.config.grsim_host, self.config.grsim_port))

        except Exception as e:
            logger.error(f"Erro ao enviar comandos UDP: {e}")

    def step(self) -> bool:
        """Executa um passo de inferencia."""
        try:
            # 1. Receber dados do SSL-Vision
            frame = self._receive_vision_data()

            # 2. Construir observações reais a partir do frame
            observations = self._build_observations_from_frame(frame)
            self.stacked_obs.update(observations)

            # 3. Computar acoes
            actions = self._compute_actions(self.stacked_obs)

            # 4. Atualizar ultimas acoes
            self.last_actions.update(actions)

            # 5. Enviar comandos UDP para grSim
            self._send_commands_udp(actions)

            self.step_count += 1

            if self.step_count % 100 == 0:
                logger.info(
                    f"Steps: {self.step_count}, Vision packets: {self.vision_packets_received}"
                )

            return True

        except Exception as e:
            logger.error(f"Erro no step {self.step_count}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_episode(self, max_steps: Optional[int] = None) -> Dict:
        """Executa um episodio completo."""
        max_steps = max_steps or (self.config.fps * 60)
        step = 0

        logger.info(f"Iniciando episodio (max_steps={max_steps})")

        try:
            while step < max_steps:
                success = self.step()
                if not success:
                    logger.warning(f"Step {step} falhou")

                step += 1
                time.sleep(1.0 / self.config.fps)

        except KeyboardInterrupt:
            logger.info(f"Episodio interrompido apos {step} steps")

        logger.info(f"Episodio finalizado. Total: {step} steps, Vision: {self.vision_packets_received} packets")

        return {
            "steps": step,
            "vision_packets": self.vision_packets_received,
            "avg_inference_time_ms": np.mean(self.inference_times) if self.inference_times else 0
        }

    def run_continuous(self):
        """Executa episodios sequenciais ate interrupcao (Ctrl+C/SIGTERM)."""
        logger.info("Executando em modo continuo (Ctrl+C para parar)")
        episode = 0
        try:
            while True:
                episode += 1
                logger.info(f"== Iniciando episodio {episode} ==")
                self.run_episode(max_steps=None)
        except KeyboardInterrupt:
            logger.info(f"Modo continuo interrompido apos {episode} episodios")


def load_config_from_env() -> DeployConfig:
    """Carrega configuracao a partir de variaveis de ambiente."""
    checkpoint_path = os.environ.get(
        "CHECKPOINT_PATH",
        "/checkpoints/PPO_selfplay_rec/PPO_Soccer_8123a_00000_0_2024-11-11_01-01-38/checkpoint_000009"
    )

    return DeployConfig(
        checkpoint_path=checkpoint_path,
        team=os.environ.get("TEAM", "blue"),
        grsim_host=os.environ.get("GRSIM_HOST", "grsim"),
        grsim_port=int(os.environ.get("GRSIM_PORT", "20011")),
        vision_port=int(os.environ.get("VISION_PORT", "10020")),
        vision_address=os.environ.get("VISION_ADDRESS", "224.5.23.2"),
        fps=int(os.environ.get("FPS", "30")),
        device=os.environ.get("DEVICE", "cpu"),
        n_robots_blue=int(os.environ.get("N_ROBOTS_BLUE", "3")),
        n_robots_yellow=int(os.environ.get("N_ROBOTS_YELLOW", "3")),
        field_type=int(os.environ.get("FIELD_TYPE", "1"))
    )


def main():
    """Funcao principal."""
    logger.info("=" * 60)
    logger.info("Deploy de Policy RL no grSim com SSL-Vision")
    logger.info("=" * 60)

    config = load_config_from_env()

    if not Path(config.checkpoint_path).exists():
        logger.error(f"Checkpoint nao encontrado: {config.checkpoint_path}")
        sys.exit(1)

    if not SSL_VISION_PROTO_AVAILABLE:
        logger.error("SSL-Vision protobuf not available. Cannot receive vision data.")
        sys.exit(1)

    try:
        controller = GrSimVisionController(config)
        controller.run_continuous()
    except Exception as e:
        logger.exception(f"Erro durante execucao: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
