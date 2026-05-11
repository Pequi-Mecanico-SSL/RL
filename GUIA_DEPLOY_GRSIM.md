# Deploy de uma policy treinada no grSim

Este guia mostra como pegar um checkpoint salvo em [volumes/dgx_checkpoints/](volumes/dgx_checkpoints/) e rodar a policy treinada controlando os robôs no [grSim](https://github.com/RoboCup-SSL/grSim) via UDP/SSL-Vision.

> Para o contexto geral do projeto (treinamento, ambiente, recompensas), veja o [README.md](README.md). Para a estrutura interna e convenções, veja [AGENTS.md](AGENTS.md).

---

## Visão geral da pipeline

Três containers Docker em `network_mode: host`:

```
grsim_simulator    -> simula o jogo, manda SSL-Vision multicast (224.5.23.2:10020)
                                     recebe comandos UDP (127.0.0.1:20011)

rl_policy_deploy           -> blue   -> recebe vision, infere policy, manda comandos
rl_policy_deploy_yellow    -> yellow -> mesmo checkpoint, espelhado em X
```

A policy é carregada de `<checkpoint>/policies/policy_blue/policy_state.pkl` por [deploy_policy_grsim.py](deploy_policy_grsim.py) (PyTorch puro, sem Ray runtime).

---

## Pré-requisitos

- Docker + Docker Compose **v2** (`docker compose`, não `docker-compose` v1).
- Linux com servidor X11 (`echo $DISPLAY` deve mostrar algo como `:0` ou `:1`) **caso queira a janela 3D nativa**.
- (opcional) GPU NVIDIA com driver instalado e `nvidia-container-toolkit` para renderização 3D acelerada.
- Submódulo do `rSoccer` clonado:
  ```bash
  git submodule update --init --recursive
  ```

---

## Modos disponíveis

| Modo | Arquivo Compose | Quando usar |
|---|---|---|
| **X11 (janela 3D nativa)** | [docker-compose.grsim-x11.yml](docker-compose.grsim-x11.yml) | Linux com X11 + GPU. Melhor visualização. |
| **VNC** | [docker-compose.grsim-vnc.yml](docker-compose.grsim-vnc.yml) | Servidor remoto / sem X11 local. Conecta via cliente VNC. |
| **Headless** | [docker-compose.grsim.yml](docker-compose.grsim.yml) | Servidor sem display. Use [scripts/vision_viewer.py](scripts/vision_viewer.py) para um top-down 2D. |

---

## Modo 1 — X11 (janela 3D nativa, recomendado em desktop Linux)

```bash
# 1. Permitir o container acessar o seu X11
xhost +local:docker

# 2. Subir tudo (grSim + 2 policies)
docker compose -f docker-compose.grsim-x11.yml up -d --build

# 3. A janela do grSim deve abrir no seu desktop.
#    Veja logs da policy ao lado:
docker logs -f rl_policy_deploy

# 4. Parar:
docker compose -f docker-compose.grsim-x11.yml down
```

Trocar o checkpoint:
```bash
CHECKPOINT_PATH=/checkpoints/PPO_selfplay_rec/<run>/checkpoint_NNNNNN \
  docker compose -f docker-compose.grsim-x11.yml up -d
```

> O caminho `CHECKPOINT_PATH` é **interno ao container** (`/checkpoints/...`). O bind-mount mapeia `volumes/dgx_checkpoints/` do host para `/checkpoints/` dentro do container.

---

## Modo 2 — VNC (sem display local)

```bash
docker compose -f docker-compose.grsim-vnc.yml up -d --build

# Conecte um cliente VNC (senha: robocup):
vncviewer localhost:5900
# ou: vinagre, krdc, tigervnc-viewer, etc.

docker compose -f docker-compose.grsim-vnc.yml down
```

> Limitação conhecida: a imagem oficial do grSim em modo VNC só inicia o binário **depois** que um cliente VNC conecta. Se ver `Vision packets: 0` no log da policy, conecte o cliente VNC primeiro.

---

## Modo 3 — Headless + viewer 2D

```bash
docker compose -f docker-compose.grsim.yml up -d --build

# Em outro terminal, no host:
python scripts/vision_viewer.py
# (requer matplotlib e protobuf no Python do host)
```

O viewer lê o SSL-Vision multicast e plota um top-down com IDs, orientação e bola. Útil em servidores headless e para debug visual rápido.

---

## Listar checkpoints disponíveis

```bash
find volumes/dgx_checkpoints -maxdepth 3 -name 'checkpoint_*' -type d | sort
```

Estrutura esperada:
```
volumes/dgx_checkpoints/PPO_selfplay_rec/<run_name>/checkpoint_NNNNNN/
  ├── algorithm_state.pkl
  ├── policies/
  │   ├── policy_blue/policy_state.pkl   ← carregado pelo deploy
  │   └── policy_yellow/policy_state.pkl  (ignorado; yellow usa o blue espelhado)
  └── ...
```

---

## Validar que a integração está funcionando

```bash
# 1. Ver a policy parseando vision e gerando ações
docker logs rl_policy_deploy 2>&1 | grep -E '(Vision packets|Steps)' | tail -5

# 2. Conferir que os robôs estão de fato se movendo
docker cp scripts/validate_motion.py rl_policy_deploy:/tmp/v.py
docker exec rl_policy_deploy python3 /tmp/v.py
#   -> imprime posições em t=0 e t=4s e o delta em mm

# 3. Reposicionar robôs no centro do campo (útil para testes)
docker cp scripts/reset_grsim_positions.py rl_policy_deploy:/tmp/reset.py
docker exec rl_policy_deploy python3 /tmp/reset.py
```

Os 4 elos validados:

| # | Elo | Como confirmar |
|---|---|---|
| 1 | grSim envia SSL-Vision multicast | logs mostram `Vision packets: > 0` crescendo |
| 2 | Policy parseia obs corretas | logs mostram `obs[-77].nonzero > 0` no modo DEBUG |
| 3 | Policy gera ações distintas por robô | 3 vetores de ação diferentes para `blue_0/1/2` |
| 4 | grSim aplica os comandos | `validate_motion.py` mostra `dist > 0` |

---

## Variáveis de ambiente (compose)

| Variável | Default | O que faz |
|---|---|---|
| `CHECKPOINT_PATH` | `/checkpoints/PPO_selfplay_rec/PPO_Soccer_baseline_2025-03-16/checkpoint_000003` | Pasta do checkpoint dentro do container. |
| `TEAM` | `blue` ou `yellow` | Qual time o container controla. Yellow espelha o policy do blue em X. |
| `GRSIM_HOST` | `127.0.0.1` | Host do grSim (network host). |
| `GRSIM_PORT` | `20011` | Porta UDP de comandos do grSim. |
| `VISION_PORT` | `10020` | Porta UDP do SSL-Vision multicast. |
| `VISION_ADDRESS` | `224.5.23.2` | Grupo multicast SSL-Vision. |
| `FPS` | `30` | Frequência do loop de inferência (precisa bater com o treino). |
| `DEVICE` | `cpu` | `cpu` ou `cuda`. CPU é o suficiente para inferência 3v3. |
| `FIELD_TYPE` | `1` | 0=6v6, 1=11v11, 2=hardware. Precisa bater com o treino. |
| `N_ROBOTS_BLUE`/`N_ROBOTS_YELLOW` | `3` | Quantos robôs cada policy controla. |

---

## Troubleshooting

| Sintoma | Causa provável | Fix |
|---|---|---|
| `Vision packets: 0` constante | grSim em modo VNC sem cliente conectado, ou multicast bloqueado | Conectar VNC viewer; checar `network_mode: host`; reiniciar `docker-compose` |
| Policy carrega mas robôs ficam parados | Robôs spawnados encostados na parede | `docker exec rl_policy_deploy python3 /tmp/reset.py` (reposiciona no centro) |
| `obs[-77].nonzero=0/77` no debug | Parse do SSL-Vision falhando silenciosamente | Aumentar log para WARNING em `_parse_vision_datagram` e ver o erro real |
| `KeyError: 'ContainerConfig'` no `docker compose up` | Compose v1 sendo invocado | Usar `docker compose` (v2). [start_policy.sh](start_policy.sh) faz o fallback automático |
| `libGL error: failed to load driver: swrast` no grSim X11 | Sem GPU / driver no container | Setar `runtime: nvidia` no compose ([docker-compose.grsim-x11.yml](docker-compose.grsim-x11.yml) já faz isso) |
| `executable file not found: grsim` | Entrypoint sobrescrito errado | Usar caminho absoluto `/usr/local/bin/grSim` |

---

## Por que os robôs podem parecer "aloprados"

Em ordem de impacto:
1. **Checkpoint pouco treinado** — 5–10 iterações de PPO não são suficientes para política coordenada.
2. **Estado inicial fora da distribuição de treino** — robôs nascem em posições que a policy nunca viu. Use o `reset_grsim_positions.py`.
3. **IDs de robôs do grSim ≠ índices da policy** — grSim manda até 11 robôs por time; a policy controla só os 3 primeiros. Garanta que IDs 0,1,2 estejam ativos.
4. **Self-play "aleatório"** — com policy pouco treinada, blue e yellow se atrapalham. Pare o yellow (`docker stop rl_policy_deploy_yellow`) para isolar o blue.

---

## Estrutura dos arquivos relacionados ao deploy

- [deploy_policy_grsim.py](deploy_policy_grsim.py) — controller principal: lê SSL-Vision, infere, envia UDP. **Único entrypoint do deploy.**
- [Dockerfile.policy](Dockerfile.policy) — imagem CPU-only PyTorch + bindings protobuf.
- [docker-compose.grsim*.yml](docker-compose.grsim.yml) — três variantes (default/headless, vnc, x11).
- [scripts/model/](scripts/model/) — versão standalone do modelo (`InferenceModel`, `InferenceBetaDist`) — sem Ray.
- [scripts/sim2real/](scripts/sim2real/) — converte frame do SSL-Vision em observação igual à do treino.
- [scripts/vision_viewer.py](scripts/vision_viewer.py) — viewer 2D top-down via multicast.
- [scripts/validate_motion.py](scripts/validate_motion.py) — mede delta de posição para validar o end-to-end.
- [scripts/reset_grsim_positions.py](scripts/reset_grsim_positions.py) — reseta robôs e bola no centro.
- Bindings protobuf (`grSim_*_pb2.py`, `ssl_vision_*_pb2.py`) — gerados de `grSim/src/proto/`. Veja `/regen-protos` em [.github/prompts/](.github/prompts/).
