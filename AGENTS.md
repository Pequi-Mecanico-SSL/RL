# AGENTS.md — Pequi Mecânico SSL — RL

Reinforcement Learning (PPO self-play, multi-agent) for RoboCup SSL EntryLevel 3v3, built on top of the [`rSoccer`](https://github.com/Pequi-Mecanico-SSL/rSoccer) gym (submodule at [rSoccer/](rSoccer/)) and Ray RLlib 2.10. Targets training inside a CUDA Docker image and policy deploy against `grSim` over UDP/protobuf.

For full project context, training/eval walkthroughs, and the Docker grSim deploy pipeline, read these first and do not duplicate them here:
- [README.md](README.md) — environment description (obs space = 8×77, action space, reward shaping), training/eval workflow.
- [GUIA_EXECUCAO_GRSIM.md](GUIA_EXECUCAO_GRSIM.md) — grSim + policy Docker Compose pipeline.
- [GUIA_VNC.md](GUIA_VNC.md) — VNC access to the headless grSim container.

The repo and most docs are in **Portuguese (pt-BR)**. Match the surrounding language in comments, log messages, and docs you touch. Code identifiers stay in English.

## Repo layout (only the non-obvious parts)

- [RL_train.py](RL_train.py) — RLlib PPO training entrypoint (self-play; trains `policy_blue`, periodically syncs `policy_yellow` from blue's weights).
- [RL_eval.py](RL_eval.py) — Loads a checkpoint and runs episodes; `CHECKPOINT_PATH_*` constants are hardcoded at the top of the file.
- [RL_infer.py](RL_infer.py) — Standalone inference (no RLlib runtime) using [scripts/model/model_inferece.py](scripts/model/model_inferece.py).
- [deploy_policy.py](deploy_policy.py), [deploy_policy_grsim.py](deploy_policy_grsim.py) — Production deploy against `grSim`: subscribes to SSL-Vision multicast and emits `grSim_Commands` over UDP. Uses the standalone `InferenceModel` + `InferenceBetaDist`, not RLlib.
- [config.yaml](config.yaml) — Single source of truth for PPO hyperparams, env config (`init_pos`, `field_type`, `stack_size=8`, `fps=30`, `match_time=40`), `custom_model.fcnet_hiddens`, and evaluation settings.
- [observations.py](observations.py), [rewards.py](rewards.py) — Observation features (positions, orientations, distances, angles, timestep, last actions) and reward terms (`r_speed`, `r_dist`, `r_off`, `r_def` combined as `0.7/0.1/0.1/0.1`; sparse goal=±10, outside=−10). Both are picklable and live at repo root because RLlib unpickles them by absolute import name (do **not** move them).
- [rSoccer/](rSoccer/) — Git submodule. The `SSLMultiAgentEnv` lives at [rSoccer/rsoccer_gym/ssl/ssl_multi_agent/ssl_multi_agent.py](rSoccer/rsoccer_gym/ssl/ssl_multi_agent/ssl_multi_agent.py). It is `pip install .`-ed into the image; editing files here only takes effect on rebuild (or with a bind mount).
- [scripts/model/](scripts/model/) — Custom RLlib model (`CustomFCNet`, optional separate VF trunk) and Beta action distributions (`TorchBetaTest_blue`/`_yellow`). A duplicate `*_inferece.py` (sic) pair exists for the deploy path that must not depend on RLlib.
- [scripts/sim2real/](scripts/sim2real/) — `frame_to_observations` and field constants used by the grSim deployer to mirror the training obs exactly.
- [scripts/gymnasium/record_video.py](scripts/gymnasium/record_video.py), [scripts/gymnasium/video_recorder.py](scripts/gymnasium/video_recorder.py) — Patched copies of `gymnasium.wrappers.*`. The Dockerfile **overwrites** the installed `site-packages` files with these; changes to recording behavior go here.
- [grSim/](grSim/) — Upstream simulator submodule (C++/Qt/ODE). We almost never modify it.
- [JuizV1/](JuizV1/) — Standalone Pygame referee/judge prototype (separate from `rSoccer/rsoccer_gym/judges/ssl_judge.py` which is what training uses).
- `*_pb2.py` at repo root — generated protobuf bindings for `grSim_Commands`, `grSim_Packet`, `grSim_Replacement`, and SSL-Vision/GC. Regenerate from the `.proto` files in [grSim/src/proto/](grSim/src/proto/); do not hand-edit.
- [volumes/dgx_checkpoints/](volumes/) — bind-mounted to `/root/ray_results/PPO_selfplay_rec` inside the training container; this is where Ray Tune writes checkpoints. `volumes/videos/` receives mp4s when training with `--evaluation`.

## Build & run

**Training image** ([Dockerfile](Dockerfile)) — CUDA 11.8 PyTorch, RLlib, rSoccer. Build from repo root: `sudo docker build -t ssl-el .`. Run with the X11 + checkpoint volume mounts from [README.md](README.md). Inside the container:
```bash
python RL_train.py --evaluation   # train + periodically record an episode
python RL_train.py                 # train only
python RL_eval.py                  # render/eval a checkpoint (edit CHECKPOINT_PATH_* first)
tensorboard --logdir=volumes/dgx_checkpoints   # run on host
```

**Deploy image** ([Dockerfile.policy](Dockerfile.policy)) — CPU-only PyTorch, no RLlib runtime needed at request time but `ray` is installed because the checkpoint pickle references RLlib classes. Orchestrated by [docker-compose.grsim.yml](docker-compose.grsim.yml) (also `.grsim-headless.yml`, `.grsim-vnc.yml`):
```bash
./start_policy.sh             # docker compose up --build (prefers v2)
./stop_policy.sh              # docker compose down
CHECKPOINT_PATH=... ./start_policy.sh   # override the checkpoint
```
Compose runs **three** containers on `network_mode: host`: `grsim_simulator`, `rl_policy_deploy` (blue), `rl_policy_deploy_yellow`. Both policy containers read the same `CHECKPOINT_PATH` env var and load `policies/policy_blue/policy_state.pkl` regardless of `TEAM`.

There is **no `pip install -e .` workflow on the host** and **no test suite**. All Python runs happen inside a container.

## Conventions and gotchas (read before changing things)

- **Use `docker compose` (v2)**, not `docker-compose` (v1). The v1 Python client throws `KeyError: 'ContainerConfig'` when recreating `rl_policy_deploy`. [start_policy.sh](start_policy.sh) handles the fallback via [scripts/sh/compose_exec.sh](scripts/sh/compose_exec.sh).
- **`packaging` is pinned `>=21.3,<23`** in [requirements.txt](requirements.txt). Newer versions break unpickling of old RLlib checkpoints (`packaging._structures` removed in 26.1; `Version` repr changed in 23+). Do not bump it.
- **`gym==0.21.0` is in `requirements.txt` for training only.** The deploy image strips it out (`grep -v "^gym==0.21.0"`) and uses `gymnasium`. New code should target `gymnasium`.
- **The PyPI package `robosim` is the wrong one.** Use `rc-robosim>=1.2.0` (still imported as `import robosim`). This is documented in [requirements.txt](requirements.txt) and assumed by [deploy_policy.py](deploy_policy.py).
- **Self-play weight sync** in [RL_train.py](RL_train.py): only `policy_blue` is in `policies_to_train`; `policy_yellow` is periodically overwritten with blue's weights via a `DefaultCallbacks` subclass. Don't add `policy_yellow` to the trainable list.
- **Observation = 8-frame stack of 77 features** built by [observations.py](observations.py) via the `StackWrapper` from [rSoccer/rsoccer_gym/Utils/Utils.py](rSoccer/rsoccer_gym/Utils/Utils.py). If you add a feature, add it both to `OBSERVATIONS` and update `obs_size` in [deploy_policy.py](deploy_policy.py) (`DeployConfig.obs_size`) — the deploy path does **not** import `OBSERVATIONS`.
- **Action distribution is a `Beta` mapped to [−1, 1]**, with per-component sign flips (`signal`) so that `yellow` mirrors `blue` on the X axis. Both [scripts/model/action_dists.py](scripts/model/action_dists.py) (training) and [scripts/model/action_dists_inferece.py](scripts/model/action_dists_inferece.py) / `InferenceBetaDist` in [deploy_policy.py](deploy_policy.py) (deploy) must be kept in sync.
- **Checkpoint layout assumption** (RLlib 2.10): `<checkpoint>/policies/policy_blue/policy_state.pkl`. The deploy loader rewrites RLlib's nested layer names (`_hidden_layers.0._model.0.weight` → `_hidden_layers.0.weight`) when copying weights into the standalone `InferenceModel`; if you change [scripts/model/custom_torch_model.py](scripts/model/custom_torch_model.py) layer naming, update [scripts/model/model_inferece.py](scripts/model/model_inferece.py) and the loader in [deploy_policy.py](deploy_policy.py) together.
- **Gymnasium recording is monkey-patched** at image build time by copying [scripts/gymnasium/record_video.py](scripts/gymnasium/record_video.py) over the installed wrapper. To debug video recording, edit those files and rebuild; editing the package files in `site-packages` won't survive.
- **Field/world parameters live in two places**: `config.yaml.env` (training) and [scripts/sim2real/config.py](scripts/sim2real/) (deploy). Keep `FIELD_LENGTH`, `FIELD_WIDTH`, `N_ROBOTS_BLUE`, `N_ROBOTS_YELLOW`, `MAX_EP_LENGTH`, `field_type` consistent across both.
- **Submodule discipline**: [rSoccer/](rSoccer/) is pinned. Clone with `git clone --recurse-submodules`. When pulling, run `git submodule update --init --recursive`. Don't commit changes inside `rSoccer/` from this repo — push them upstream first.
- **Logs**: training writes to `/root/ray_results/...` (volume → `volumes/dgx_checkpoints/`); deploy writes to `/app/logs/policy.log` (volume → [logs/](logs/)).

## Subagentes disponíveis (foco: rodar checkpoints no grSim)

O objetivo recorrente é **pegar um checkpoint em [volumes/dgx_checkpoints/](volumes/dgx_checkpoints/) e jogá-lo no grSim**. Quatro subagentes especializados cobrem os eixos da integração — invoque em paralelo quando a tarefa toca múltiplos eixos:

| Subagente | Quando invocar | Arquivos sob escopo |
|---|---|---|
| [`grsim-deploy`](.github/agents/grsim-deploy.agent.md) | Rodar/diagnosticar um checkpoint contra o grSim, carregar `policy_state.pkl`, loop UDP, troca de TEAM, mismatch obs/action. | [deploy_policy_grsim.py](deploy_policy_grsim.py), [deploy_policy.py](deploy_policy.py), [scripts/model/model_inferece.py](scripts/model/model_inferece.py), [scripts/model/action_dists_inferece.py](scripts/model/action_dists_inferece.py) |
| [`sim2real`](.github/agents/sim2real.agent.md) | Paridade entre `OBSERVATIONS` (treino) e `frame_to_observations` (deploy). Qualquer mudança em features. | [observations.py](observations.py), [scripts/sim2real/](scripts/sim2real/) |
| [`grsim-proto`](.github/agents/grsim-proto.agent.md) | Editar `.proto` ou regenerar `*_pb2.py`. | [grSim/src/proto/](grSim/src/proto/), `*_pb2.py` no root, [Dockerfile.policy](Dockerfile.policy) |
| [`docker-grsim`](.github/agents/docker-grsim.agent.md) | Subir/derrubar/depurar pipeline Compose, VNC, network_mode host. | [docker-compose.grsim*.yml](docker-compose.grsim.yml), [Dockerfile.policy](Dockerfile.policy), [start_policy.sh](start_policy.sh)/[stop_policy.sh](stop_policy.sh) |

Instruções sempre-on:
- [.github/instructions/deploy-parity.instructions.md](.github/instructions/deploy-parity.instructions.md) — twins treino↔deploy.
- [.github/instructions/protobuf.instructions.md](.github/instructions/protobuf.instructions.md) — regras de `*.proto` e `*_pb2.py`.

Prompts slash:
- `/run-checkpoint-grsim` — workflow guiado para subir um checkpoint no grSim ([.github/prompts/run-checkpoint-grsim.prompt.md](.github/prompts/run-checkpoint-grsim.prompt.md)).
- `/regen-protos` — regenera os `*_pb2.py` ([.github/prompts/regen-protos.prompt.md](.github/prompts/regen-protos.prompt.md)).

**Padrão de orquestração**: ao mudar uma feature de observação, o agente padrão dispara `sim2real` + `grsim-deploy` em paralelo (relatórios), consolida o patch, e só então edita. `docker-grsim` e `grsim-proto` ficam de fora a menos que o impacto chegue na imagem/protos.

## When in doubt

- Reproduce the failure inside the matching container (`ssl-el` for training, `rl_policy_deploy` for deploy) before suspecting a code bug — most issues are environment/version mismatches.
- For pickle / `ModuleNotFoundError` on checkpoint load: check `PYTHONPATH=/app:/app/rSoccer` (deploy) or that `rewards.py`/`observations.py` are at the path RLlib pickled them from.
- For UDP / multicast issues with grSim: see [scripts/check_grsim_multicast_recv.py](scripts/check_grsim_multicast_recv.py) and [scripts/sh/check_grsim_vision_udp.sh](scripts/sh/check_grsim_vision_udp.sh).
