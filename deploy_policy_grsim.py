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
    ACTION_MODE: Selecao da acao Beta ('mean' ou 'sample')
    ACTION_SEED: Seed do gerador PyTorch para amostragem reproduzivel
"""

import os
import sys
import pickle
import time
import logging
import socket
import struct
import signal
import threading
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

# Contrato historico de observacao usado pelo baseline de 2025.
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
try:
    from sim2real.state_to_obs import frame_to_observations
    OBSERVATIONS_AVAILABLE = True
    print("Observations module loaded successfully")
except ImportError as e:
    OBSERVATIONS_AVAILABLE = False
    print(f"Warning: observations module not available: {e}")

from scripts.model.model_inferece import InferenceModel
from scripts.model.action_dists_inferece import InferenceBetaDist
from scripts.reset_grsim_positions import (
    BLUE as KICKOFF_BLUE,
    YELLOW as KICKOFF_YELLOW,
    perform_kickoff,
)

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
    match_time: int = 40
    vision_stale_timeout: float = 0.25
    action_mode: str = "mean"
    action_seed: int = 0
    episode_reset: bool = False
    kickoff_master: bool = False

    def __post_init__(self):
        if self.action_mode not in {"mean", "sample"}:
            raise ValueError(
                f"ACTION_MODE invalido: {self.action_mode!r}; use 'mean' ou 'sample'"
            )


def select_beta_action(distribution, action_mode):
    """Seleciona a media ou uma amostra da Beta conforme o modo operacional."""
    if action_mode == "mean":
        return distribution.deterministic_sample()
    if action_mode == "sample":
        return distribution.sample()
    raise ValueError(f"action_mode invalido: {action_mode!r}")


def is_kickoff_formation(world_state, ball_tolerance=0.15, robot_tolerance=0.25):
    """Detecta a assinatura fisica do kickoff enviada pelo replacement.

    A combinacao bola no centro + seis robos na formacao inicial nao ocorre
    durante o jogo, entao serve como sinal de reset para os dois containers
    sem canal de comunicacao adicional.
    """
    ball = world_state.get("ball")
    if ball is None or np.hypot(ball[0], ball[1]) > ball_tolerance:
        return False
    for team_key, formation in (("robots_blue", KICKOFF_BLUE),
                                ("robots_yellow", KICKOFF_YELLOW)):
        robots = world_state.get(team_key, {})
        for robot_id, (x, y, _dir) in enumerate(formation):
            pose = robots.get(f"robot_{robot_id}")
            if pose is None or np.hypot(pose[0] - x, pose[1] - y) > robot_tolerance:
                return False
    return True


def normalized_action_to_grsim(action, theta_degrees):
    """Converte acao global normalizada para velocidades locais do grSim."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    global_velocity = action[:2] * 1.5
    speed = np.linalg.norm(global_velocity)
    if speed > 1.5:
        global_velocity *= 1.5 / speed

    theta = np.deg2rad(theta_degrees)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    tangent = global_velocity[0] * cos_theta + global_velocity[1] * sin_theta
    normal = -global_velocity[0] * sin_theta + global_velocity[1] * cos_theta
    # O baseline aplicava kick fixo de 3 m/s para qualquer acao positiva.
    return tangent, normal, action[2] * 10.0, 3.0 if action[3] > 0.0 else 0.0


class GrSimVisionController:
    """Controlador que recebe vision SSL e envia comandos UDP para o grSim."""

    def __init__(self, config: DeployConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.action_seed)

        logger.info(f"Inicializando GrSimVisionController:")
        logger.info(f"  Checkpoint: {config.checkpoint_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Team: {config.team}")
        logger.info(f"  Action mode: {config.action_mode} (seed={config.action_seed})")
        logger.info(
            f"  Episode reset: {config.episode_reset} "
            f"(kickoff_master={config.kickoff_master})"
        )
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
        self.world_state = {
            "robots_blue": {}, "robots_yellow": {}, "ball": None
        }
        self.entity_updated_at = {}
        self.stop_event = threading.Event()
        self._stale_logged = False
        self._vision_first_packet_logged = False
        self._kickoff_latched = False
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

        model.load_state_dict(weights_dict, strict=True)
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

        if wrapper.geometry.field:
            field = wrapper.geometry.field
            self.field_info = {
                "length": field.field_length / 1000.0,
                "width": field.field_width / 1000.0,
                "goal_width": field.goal_width / 1000.0 if field.goal_width else 1.0,
            }

        detection = wrapper.detection
        if not (detection.robots_blue or detection.robots_yellow or detection.balls):
            return None
        now = time.monotonic()

        for robot in detection.robots_blue:
            key = f"robot_{robot.robot_id}"
            self.world_state["robots_blue"][key] = [
                robot.x / 1000.0, robot.y / 1000.0, np.rad2deg(robot.orientation)
            ]
            self.entity_updated_at[("blue", robot.robot_id)] = now

        for robot in detection.robots_yellow:
            key = f"robot_{robot.robot_id}"
            self.world_state["robots_yellow"][key] = [
                robot.x / 1000.0, robot.y / 1000.0, np.rad2deg(robot.orientation)
            ]
            self.entity_updated_at[("yellow", robot.robot_id)] = now

        if detection.balls:
            ball = detection.balls[0]
            self.world_state["ball"] = [ball.x / 1000.0, ball.y / 1000.0]
            self.entity_updated_at[("ball", 0)] = now

        return self.world_state

    def _receive_vision_data(self) -> Optional[Dict]:
        """Drena a fila SSL-Vision e retorna o snapshot persistente mesclado."""
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

    def _vision_is_fresh(self) -> bool:
        now = time.monotonic()
        required = [("ball", 0)]
        required += [("blue", i) for i in range(self.config.n_robots_blue)]
        required += [("yellow", i) for i in range(self.config.n_robots_yellow)]
        return all(
            key in self.entity_updated_at
            and now - self.entity_updated_at[key] <= self.config.vision_stale_timeout
            for key in required
        )

    def _build_observations_from_frame(self, frame: Dict) -> Dict[str, np.ndarray]:
        """Constroi o stack pelo contrato historico exato do baseline."""
        if frame is None or not OBSERVATIONS_AVAILABLE:
            # Fallback: manter observacoes anteriores
            return self.stacked_obs

        try:
            observations = frame_to_observations(
                frame, self.last_actions, self.stacked_obs, steps=self.step_count
            )
        except Exception as e:
            logger.warning(f"Falha ao computar features de observacao: {e}")
            return self.stacked_obs
        for robot_key, obs in observations.items():
            if obs.shape != (self.config.obs_size * self.config.n_stack,):
                raise ValueError(f"Observacao {robot_key} tem shape invalido: {obs.shape}")
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
        if not torch.isfinite(model_input).all():
            raise ValueError("Observacao contem NaN/Inf")

        with torch.no_grad():
            model_output, value = self.model(model_input)
        if not torch.isfinite(model_output).all():
            raise ValueError("Logits contem NaN/Inf")

        signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
        distribution = InferenceBetaDist(model_output, signal=signal)

        actions_tensor = select_beta_action(distribution, self.config.action_mode)
        if not torch.isfinite(actions_tensor).all():
            raise ValueError("Acoes contem NaN/Inf")
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

            for robot_name, action in actions.items():
                color, idx_str = robot_name.split("_")
                idx = int(idx_str)

                if color != self.config.team:
                    continue

                robot_cmd = commands.robot_commands.add()
                robot_cmd.id = idx
                robot_cmd.wheelsspeed = False
                pose = self.world_state[f"robots_{color}"].get(
                    f"robot_{idx}", [0.0, 0.0, 0.0]
                )
                tangent, normal, angular, kick = normalized_action_to_grsim(action, pose[2])
                robot_cmd.veltangent = tangent
                robot_cmd.velnormal = normal
                robot_cmd.velangular = angular
                robot_cmd.kickspeedx = kick
                robot_cmd.kickspeedz = 0.0
                robot_cmd.spinner = False

            data = packet.SerializeToString()
            self.command_socket.sendto(data, (self.config.grsim_host, self.config.grsim_port))

        except Exception as e:
            logger.error(f"Erro ao enviar comandos UDP: {e}")

    def _send_zero_commands(self):
        zeros = {
            f"{self.config.team}_{i}": [0.0, 0.0, 0.0, 0.0]
            for i in range(
                self.config.n_robots_blue if self.config.team == "blue"
                else self.config.n_robots_yellow
            )
        }
        self._send_commands_udp(zeros)

    def reset_episode(self):
        self.step_count = 0
        self.stacked_obs = self._init_stacked_obs()
        self.last_actions = self._init_actions()

    def step(self) -> bool:
        """Executa um passo de inferencia."""
        try:
            # 1. Receber dados do SSL-Vision
            frame = self._receive_vision_data()

            if not self._vision_is_fresh():
                if not self._stale_logged:
                    logger.warning("Visao incompleta/stale; enviando comandos zero")
                    self._stale_logged = True
                self._send_zero_commands()
                return True
            self._stale_logged = False

            # Reset temporal sincronizado: a formacao de kickoff so aparece
            # apos o replacement, entao blue e yellow reiniciam juntos.
            if self.config.episode_reset:
                if is_kickoff_formation(self.world_state):
                    if not self._kickoff_latched:
                        self._kickoff_latched = True
                        logger.info(
                            "Kickoff detectado; reiniciando estado temporal do episodio"
                        )
                        self.reset_episode()
                else:
                    self._kickoff_latched = False

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
                    f"Steps: {self.step_count}, Vision packets: {self.vision_packets_received}, "
                    f"Actions: {actions}"
                )

            return True

        except Exception as e:
            logger.error(f"Erro no step {self.step_count}: {e}")
            self._send_zero_commands()
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_episode(self, max_steps: Optional[int] = None, reset_state: bool = True) -> Dict:
        """Executa um episodio completo."""
        max_steps = max_steps or (self.config.fps * self.config.match_time)
        step = 0
        if reset_state:
            self.reset_episode()

        logger.info(f"Iniciando episodio (max_steps={max_steps})")

        try:
            while step < max_steps and not self.stop_event.is_set():
                success = self.step()
                if not success:
                    logger.warning(f"Step {step} falhou")

                step += 1
                self.stop_event.wait(1.0 / self.config.fps)

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
            while not self.stop_event.is_set():
                episode += 1
                logger.info(f"== Iniciando episodio {episode} ==")
                if self.config.episode_reset and self.config.kickoff_master:
                    # Reproduz a estrutura de episodios do treino: replacement
                    # de kickoff a cada janela de match_time.
                    logger.info("Enviando kickoff fisico (master)")
                    try:
                        perform_kickoff(
                            self.command_socket,
                            (self.config.grsim_host, self.config.grsim_port),
                        )
                    except Exception as e:
                        logger.error(f"Falha ao enviar kickoff: {e}")
                    self.reset_episode()
                    self.run_episode(max_steps=None, reset_state=False)
                elif self.config.episode_reset:
                    # Nao-master: o reset temporal vem da deteccao do kickoff.
                    self.run_episode(max_steps=None, reset_state=False)
                else:
                    # O mundo fisico nao e reposicionado entre janelas de logging.
                    # Zerar stack/last_actions aqui criaria um estado artificial.
                    self.run_episode(max_steps=None, reset_state=(episode == 1))
        except KeyboardInterrupt:
            logger.info(f"Modo continuo interrompido apos {episode} episodios")
        finally:
            for _ in range(3):
                self._send_zero_commands()
            if self.vision_socket:
                self.vision_socket.close()
            self.command_socket.close()

    def request_stop(self, signum=None, _frame=None):
        logger.info(f"Encerramento solicitado (signal={signum})")
        self.stop_event.set()


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
        field_type=int(os.environ.get("FIELD_TYPE", "1")),
        match_time=int(os.environ.get("MATCH_TIME", "40")),
        vision_stale_timeout=float(os.environ.get("VISION_STALE_TIMEOUT", "0.25")),
        action_mode=os.environ.get("ACTION_MODE", "mean").strip().lower(),
        action_seed=int(os.environ.get("ACTION_SEED", "0")),
        episode_reset=os.environ.get("EPISODE_RESET", "0").strip() in ("1", "true"),
        kickoff_master=os.environ.get("KICKOFF_MASTER", "0").strip() in ("1", "true"),
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
        signal.signal(signal.SIGINT, controller.request_stop)
        signal.signal(signal.SIGTERM, controller.request_stop)
        controller.run_continuous()
    except Exception as e:
        logger.exception(f"Erro durante execucao: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
