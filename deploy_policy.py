#!/usr/bin/env python3
"""
Deploy de Policy RL no grSim

Este script carrega um checkpoint RLlib e executa a política
no simulador grSim usando a biblioteca rc-robosim.

Uso:
    python deploy_policy.py

Variáveis de ambiente:
    CHECKPOINT_PATH: Caminho para o checkpoint RLlib
    TEAM: Time a controlar ('blue' ou 'yellow')
    GRSIM_HOST: Host do grSim (default: 127.0.0.1)
    GRSIM_PORT: Porta de comandos UDP (default: 20011)
    FPS: Frames por segundo (default: 30)
    DEVICE: Dispositivo PyTorch ('cpu' ou 'cuda')
"""

import os
import sys
import pickle
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/policy.log') if os.path.exists('/app/logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger('deploy_policy')

# Importar componentes do repositório
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.model.model_inferece import InferenceModel

# Tentar importar rc-robosim
try:
    import robosim
    ROBOSIM_AVAILABLE = True
    logger.info("rc-robosim (robosim) importado com sucesso")
except ImportError:
    ROBOSIM_AVAILABLE = False
    logger.warning("rc-robosim (robosim) nao disponivel. Modo simulado sera usado.")

# Tentar importar funções de conversão de estado
try:
    from scripts.sim2real.state_to_obs import frame_to_observations
    from scripts.sim2real.config import (
        FIELD_LENGTH, FIELD_WIDTH, N_ROBOTS_BLUE,
        N_ROBOTS_YELLOW, MAX_EP_LENGTH, GOAL, BALL, ROBOT
    )
    STATE_CONV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Funcoes de conversao de estado nao disponiveis: {e}")
    STATE_CONV_AVAILABLE = False


@dataclass
class DeployConfig:
    """Configuracao para deploy."""
    checkpoint_path: str
    team: str = "blue"
    grsim_host: str = "127.0.0.1"
    grsim_port: int = 20011
    vision_port: int = 10002
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
    """Distribuicao Beta para amostragem de acoes (versao standalone)."""

    def __init__(self, inputs: torch.Tensor, signal: List[float] = None):
        self.inputs = inputs
        # Estabilizar parametros
        self.inputs = torch.clamp(self.inputs, -20, 20)
        self.inputs = torch.log(torch.exp(self.inputs) + 1.0) + 1.0

        # Dividir em alpha e beta
        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        self.signal = torch.tensor(signal or [1, 1, 1, 1], dtype=torch.float32)

    def sample(self) -> torch.Tensor:
        """Amostra da distribuicao Beta."""
        sample = self.dist.rsample()
        # Mapear de [0,1] para [-1,1]
        sample = sample * 2.0 - 1.0
        return self.signal.to(sample.device) * sample

    def mean(self) -> torch.Tensor:
        """Media da distribuicao (acao deterministica)."""
        mean = self.dist.mean
        mean = mean * 2.0 - 1.0
        return self.signal.to(mean.device) * mean


class GrSimDeployer:
    """
    Classe principal para deploy de policies no grSim.
    """

    def __init__(self, config: DeployConfig):
        """
        Inicializa o deployer.

        Args:
            config: Configuracao de deploy
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        logger.info(f"Inicializando GrSimDeployer:")
        logger.info(f"  Checkpoint: {config.checkpoint_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Team: {config.team}")
        logger.info(f"  Robots: {config.n_robots_blue} blue, {config.n_robots_yellow} yellow")

        # Carregar modelo
        self.model = self._load_model()
        self.model.eval()

        # Inicializar buffer de observações
        self.stacked_obs = self._init_stacked_obs()
        self.last_actions = self._init_actions()

        # Inicializar simulador (se disponivel)
        self.sim = self._init_simulator() if ROBOSIM_AVAILABLE else None

        # Contadores para estatisticas
        self.step_count = 0
        self.inference_times = []

    def _load_model(self) -> InferenceModel:
        """Carrega o modelo do checkpoint RLlib."""
        model = InferenceModel(
            input_size=self.config.obs_size * self.config.n_stack,
            output_size=2 * self.config.action_size  # 2 robos x 4 acoes
        )

        policy_file = Path(self.config.checkpoint_path) / "policies/policy_blue/policy_state.pkl"

        if not policy_file.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado: {policy_file}")

        logger.info(f"Carregando checkpoint: {policy_file}")

        with open(policy_file, "rb") as f:
            policy_state = pickle.load(f)

        # Converter pesos do formato RLlib para PyTorch
        weights_dict = {}
        for layer_name, weights in policy_state["weights"].items():
            split = layer_name.split(".")
            if "_logits" == split[0] or "_value_branch" == split[0]:
                new_layer_name = split[0] + "." + split[-1]
            else:
                # Mapear camadas ocultas
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

    def _ssl_initial_placement(
        self,
    ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        """
        Bola e poses iniciais dos robos (SSL): mesmos dados para SSL() e reset().
        Listas Python puras — pybind11 nao aceita kwargs e prefere tipos explicitos.
        """
        ball_pos: List[float] = [0.0, 0.0, 0.0, 0.0]
        blue_tpl = [[-1.5, 0.0, 0.0], [-2.0, 1.0, 0.0], [-2.0, -1.0, 0.0]]
        yellow_tpl = [[1.5, 0.0, 180.0], [2.0, 1.0, 180.0], [2.0, -1.0, 180.0]]
        nb = self.config.n_robots_blue
        ny = self.config.n_robots_yellow

        def expand(
            template: List[List[float]], n: int, fallback: Tuple[float, float, float]
        ) -> List[List[float]]:
            rows: List[List[float]] = []
            for i in range(n):
                if i < len(template):
                    x, y, th = template[i]
                    rows.append([float(x), float(y), float(th)])
                else:
                    rows.append(
                        [
                            float(fallback[0] * (i + 1)),
                            float(fallback[1]),
                            float(fallback[2]),
                        ]
                    )
            return rows

        blue_pos = expand(blue_tpl, nb, (-0.2, 0.0, 0.0))
        yellow_pos = expand(yellow_tpl, ny, (0.2, 0.0, 180.0))
        return ball_pos, blue_pos, yellow_pos

    def _init_simulator(self):
        """Inicializa o simulador grSim via rc-robosim."""
        if not ROBOSIM_AVAILABLE:
            logger.warning("rc-robosim nao disponivel, simulador nao inicializado")
            return None

        try:
            ball_pos, blue_pos, yellow_pos = self._ssl_initial_placement()

            # rc-robosim/pybind11: SSL() so aceita argumentos posicionais
            sim = robosim.SSL(
                self.config.field_type,
                self.config.n_robots_blue,
                self.config.n_robots_yellow,
                int(1000 / self.config.fps),
                ball_pos,
                blue_pos,
                yellow_pos,
            )

            logger.info("Simulador rc-robosim inicializado")
            return sim
        except Exception as e:
            logger.error(f"Erro ao inicializar simulador: {e}")
            return None

    def _get_frame_state(self) -> Optional[Dict]:
        """Obtem o estado atual do simulador."""
        if self.sim is None:
            return None

        try:
            state = self.sim.get_state()

            # Converter formato rc-robosim para formato do repositorio
            # Formato esperado: [ball_x, ball_y, ball_vx, ball_vy, blue_robots..., yellow_robots...]
            frame = {
                "robots_blue": {},
                "robots_yellow": {},
                "ball": [state[0], state[1]]  # x, y da bola
            }

            # Extrair posicoes dos robos
            idx = 4  # Offset apos bola (x, y, vx, vy)
            for i in range(self.config.n_robots_blue):
                frame["robots_blue"][f"robot_{i}"] = [
                    state[idx],      # x
                    state[idx + 1],  # y
                    state[idx + 2]   # theta
                ]
                idx += 3

            for i in range(self.config.n_robots_yellow):
                frame["robots_yellow"][f"robot_{i}"] = [
                    state[idx],
                    state[idx + 1],
                    state[idx + 2]
                ]
                idx += 3

            return frame
        except Exception as e:
            logger.error(f"Erro ao obter estado: {e}")
            return None

    def _convert_to_observations(self, frame: Dict, step: int) -> Dict[str, np.ndarray]:
        """Converte frame do simulador para observacoes."""
        if STATE_CONV_AVAILABLE and frame is not None:
            try:
                # Usar funcao do repositorio se disponivel
                self.stacked_obs = frame_to_observations(
                    frame, self.last_actions, self.stacked_obs
                )
                return self.stacked_obs
            except Exception as e:
                logger.warning(f"Erro na conversao de estado: {e}, usando fallback")

        # Fallback: gerar observacoes dummy para teste
        for key in self.stacked_obs.keys():
            # Shift das observacoes
            self.stacked_obs[key] = np.roll(self.stacked_obs[key], -self.config.obs_size)
            # Nova observacao (dummy ou ruido para teste)
            new_obs = np.random.randn(self.config.obs_size).astype(np.float32) * 0.1
            self.stacked_obs[key][-self.config.obs_size:] = new_obs

        return self.stacked_obs

    def _compute_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """Computa acoes usando o modelo."""
        start_time = time.time()

        # Preparar input para o modelo (apenas robos do time controlado)
        model_inputs = []
        robot_names = []

        team = self.config.team
        for robot_name in sorted(observations.keys()):
            if team in robot_name:
                model_inputs.append(observations[robot_name])
                robot_names.append(robot_name)

        if not model_inputs:
            return {}

        # Executar modelo
        model_input = torch.tensor(
            np.array(model_inputs, dtype=np.float32)
        ).to(self.device)

        with torch.no_grad():
            model_output, value = self.model(model_input)

        # Aplicar distribuicao Beta
        signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
        distribution = InferenceBetaDist(model_output, signal=signal)

        # Usar media para acao deterministica (ou sample para estocastica)
        actions_tensor = distribution.mean()
        actions = actions_tensor.detach().cpu().numpy()

        # Mapear acoes para robos
        action_dict = {}
        for i, robot_name in enumerate(robot_names):
            action_dict[robot_name] = actions[i].tolist()

        # Registrar tempo de inferencia
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)

        if len(self.inference_times) >= 100:
            avg_time = np.mean(self.inference_times[-100:])
            logger.debug(f"Tempo medio de inferencia: {avg_time:.2f}ms")

        return action_dict

    def _send_commands(self, actions: Dict[str, List[float]]):
        """Envia comandos para o simulador."""
        if self.sim is None:
            return

        try:
            # Converter acoes [-1, 1] para comandos de velocidade
            max_linear = 5.0  # m/s
            max_angular = 20.0  # rad/s
            kick_speed = 3.0  # m/s

            # Preparar comandos para todos os robos
            total_robots = self.config.n_robots_blue + self.config.n_robots_yellow
            commands = np.zeros((total_robots, 8), dtype=np.float64)

            # Preencher comandos do time controlado
            for robot_name, action in actions.items():
                color, idx_str = robot_name.split("_")
                idx = int(idx_str)

                if color == "blue":
                    rbt_id = idx
                else:
                    rbt_id = self.config.n_robots_blue + idx

                v_x = action[0] * max_linear
                v_y = action[1] * max_linear
                v_theta = action[2] * max_angular
                kick = 1.0 if action[3] > 0.5 else 0.0

                # Formato: [wheel_speed_flag, v_x, v_y, v_theta, wheel3, kick_v_x, kick_v_z, dribbler]
                commands[rbt_id] = [
                    0,           # wheel_speed (0 = usar velocidades globais)
                    v_x,         # v_x
                    v_y,         # v_y
                    v_theta,     # v_theta
                    0,           # reservado
                    kick * kick_speed,  # kick_v_x
                    0,           # kick_v_z (chip kick)
                    0            # dribbler
                ]

            # Enviar para simulador (array C-contiguo float64, como em RSimSSL)
            self.sim.step(np.ascontiguousarray(commands, dtype=np.float64))

        except Exception as e:
            logger.error(f"Erro ao enviar comandos: {e}")

    def step(self) -> bool:
        """
        Executa um passo de inferencia.

        Returns:
            True se sucesso, False caso contrario
        """
        try:
            # 1. Obter estado atual
            frame = self._get_frame_state()

            # 2. Converter para observacoes
            observations = self._convert_to_observations(frame, self.step_count)

            # 3. Computar acoes
            actions = self._compute_actions(observations)

            # 4. Atualizar ultimas acoes
            self.last_actions.update(actions)

            # 5. Enviar comandos para simulador
            self._send_commands(actions)

            self.step_count += 1

            # Log periodico
            if self.step_count % 100 == 0:
                logger.info(f"Steps executados: {self.step_count}")

            return True

        except Exception as e:
            logger.error(f"Erro no step {self.step_count}: {e}")
            return False

    def run_episode(self, max_steps: Optional[int] = None) -> Dict:
        """
        Executa um episodio completo.

        Args:
            max_steps: Numero maximo de steps (None = ilimitado)

        Returns:
            Estatisticas do episodio
        """
        max_steps = max_steps or (self.config.fps * 60)  # 60 segundos padrao
        step = 0

        logger.info(f"Iniciando episodio (max_steps={max_steps})")

        # Resetar simulador se disponivel (reset exige ball + blue + yellow, como em rsim.py)
        if self.sim:
            try:
                b0, bl0, yl0 = self._ssl_initial_placement()
                self.sim.reset(b0, bl0, yl0)
            except Exception as e:
                logger.warning(f"Nao foi possivel resetar simulador: {e}")

        # Resetar buffers
        self.stacked_obs = self._init_stacked_obs()
        self.last_actions = self._init_actions()

        try:
            while step < max_steps:
                success = self.step()
                if not success:
                    logger.warning(f"Step {step} falhou, continuando...")

                step += 1

                # Controle de frequencia (FPS)
                time.sleep(1.0 / self.config.fps)

        except KeyboardInterrupt:
            logger.info(f"Episodio interrompido pelo usuario apos {step} steps")

        logger.info(f"Episodio finalizado. Total de steps: {step}")

        return {
            "steps": step,
            "avg_inference_time_ms": np.mean(self.inference_times) if self.inference_times else 0,
            "final_actions": self.last_actions
        }

    def run_continuous(self):
        """Executa continuamente ate interrupcao."""
        logger.info("Executando em modo continuo (Ctrl+C para parar)")
        self.run_episode(max_steps=None)


def load_config_from_env() -> DeployConfig:
    """Carrega configuracao a partir de variaveis de ambiente."""
    checkpoint_path = os.environ.get(
        "CHECKPOINT_PATH",
        "/checkpoints/PPO_selfplay_rec/PPO_Soccer_baseline_2025-03-16/checkpoint_000003"
    )

    return DeployConfig(
        checkpoint_path=checkpoint_path,
        team=os.environ.get("TEAM", "blue"),
        grsim_host=os.environ.get("GRSIM_HOST", "127.0.0.1"),
        grsim_port=int(os.environ.get("GRSIM_PORT", "20011")),
        vision_port=int(os.environ.get("VISION_PORT", "10002")),
        vision_address=os.environ.get("VISION_ADDRESS", "224.5.23.2"),
        fps=int(os.environ.get("FPS", "30")),
        device=os.environ.get("DEVICE", "cpu"),
        n_robots_blue=int(os.environ.get("N_ROBOTS_BLUE", "3")),
        n_robots_yellow=int(os.environ.get("N_ROBOTS_YELLOW", "3")),
        field_type=int(os.environ.get("FIELD_TYPE", "1"))
    )


def test_checkpoint_loading(checkpoint_path: str) -> bool:
    """Testa se um checkpoint pode ser carregado."""
    try:
        logger.info(f"Testando carregamento do checkpoint: {checkpoint_path}")

        policy_file = Path(checkpoint_path) / "policies/policy_blue/policy_state.pkl"
        if not policy_file.exists():
            logger.error(f"Arquivo de policy nao encontrado: {policy_file}")
            return False

        with open(policy_file, "rb") as f:
            policy_state = pickle.load(f)

        logger.info(f"Checkpoint carregado com sucesso!")
        logger.info(f"  Chaves disponiveis: {list(policy_state.keys())}")

        if "weights" in policy_state:
            logger.info(f"  Numero de camadas: {len(policy_state['weights'])}")
            for key in list(policy_state["weights"].keys())[:5]:
                logger.info(f"    - {key}: {policy_state['weights'][key].shape}")

        return True

    except Exception as e:
        logger.error(f"Erro ao carregar checkpoint: {e}")
        return False


def main():
    """Funcao principal."""
    logger.info("=" * 60)
    logger.info("Deploy de Policy RL no grSim")
    logger.info("=" * 60)

    # Verificar argumentos de linha de comando
    if len(sys.argv) > 1 and sys.argv[1] == "--test-checkpoint":
        checkpoint = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CHECKPOINT_PATH")
        success = test_checkpoint_loading(checkpoint)
        sys.exit(0 if success else 1)

    # Carregar configuracao
    config = load_config_from_env()

    # Verificar se checkpoint existe
    if not Path(config.checkpoint_path).exists():
        logger.error(f"Checkpoint nao encontrado: {config.checkpoint_path}")
        logger.error("Verifique o caminho ou ajuste a variavel CHECKPOINT_PATH")
        sys.exit(1)

    # Listar checkpoints disponiveis
    checkpoints_base = Path(config.checkpoint_path).parent.parent
    if checkpoints_base.exists():
        logger.info("Checkpoints disponiveis:")
        for checkpoint_dir in sorted(checkpoints_base.glob("**/checkpoint_*")):
            policy_file = checkpoint_dir / "policies/policy_blue/policy_state.pkl"
            status = "OK" if policy_file.exists() else "FALTANDO"
            logger.info(f"  - {checkpoint_dir}: {status}")

    try:
        # Criar deployer
        deployer = GrSimDeployer(config)

        # Executar
        if ROBOSIM_AVAILABLE and deployer.sim is not None:
            deployer.run_continuous()
        else:
            logger.warning("Simulador nao disponivel, executando em modo de teste")
            # Modo de teste: apenas verificar se o modelo funciona
            for i in range(10):
                deployer.step()
                time.sleep(0.1)
            logger.info("Modo de teste concluido")

    except FileNotFoundError as e:
        logger.error(f"Arquivo nao encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro durante execucao: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
