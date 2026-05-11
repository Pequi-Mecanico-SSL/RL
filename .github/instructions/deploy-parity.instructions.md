---
description: "Paridade obrigatória entre código de treino e código de deploy do grSim. Sempre considerar ao editar observations.py, rewards.py, scripts/model/**, scripts/sim2real/**, deploy_policy*.py."
applyTo: "observations.py, rewards.py, scripts/model/**, scripts/sim2real/**, deploy_policy*.py"
---

# Paridade Treino ↔ Deploy

O deploy contra o grSim **não usa RLlib em runtime**. Existe um "twin" standalone para cada componente do agente. Manter os dois lados em sincronia é a fonte mais comum de bugs silenciosos.

## Pares obrigatórios

| Treino (RLlib) | Deploy (standalone) | O que precisa bater |
|---|---|---|
| [observations.py](../../observations.py) `OBSERVATIONS` | [scripts/sim2real/state_to_obs.py](../../scripts/sim2real/state_to_obs.py) `frame_to_observations` | Ordem, normalização, kwargs, total = 77 features |
| [scripts/model/custom_torch_model.py](../../scripts/model/custom_torch_model.py) `CustomFCNet` | [scripts/model/model_inferece.py](../../scripts/model/model_inferece.py) `InferenceModel` | Camadas, ativação, nomes (loader em [deploy_policy.py](../../deploy_policy.py) renomeia `_hidden_layers.<i>._model.0.weight` → `_hidden_layers.<i*2>.weight`) |
| [scripts/model/action_dists.py](../../scripts/model/action_dists.py) `TorchBetaTest_*` | [scripts/model/action_dists_inferece.py](../../scripts/model/action_dists_inferece.py) + `InferenceBetaDist` em [deploy_policy.py](../../deploy_policy.py) | Softplus, mapeamento [0,1]→[-1,1], vetor `signal` (yellow espelha X) |

## Checklist ao editar

- [ ] Mudou `OBSERVATIONS`? → Espelhar em `frame_to_observations` **e** atualizar `DeployConfig.obs_size` em [deploy_policy.py](../../deploy_policy.py) no mesmo PR.
- [ ] Mudou `fcnet_hiddens` em [config.yaml](../../config.yaml)? → Confirmar que `InferenceModel` reconstrói o mesmo grafo (ele lê o shape pelo state_dict; chaves faltantes = problema).
- [ ] Mudou alguma camada com nome novo no `CustomFCNet`? → Atualizar o renomeador em [deploy_policy.py](../../deploy_policy.py) **e** o construtor do `InferenceModel`.
- [ ] Mudou a distribuição de ações? → Os DOIS arquivos `action_dists*.py`. O `signal` (sinais por componente) precisa ser idêntico.
- [ ] **Não mover** [observations.py](../../observations.py) ou [rewards.py](../../rewards.py) do root: o pickle do checkpoint os referencia como módulos top-level.

## Sinais que a paridade quebrou

- Robôs ficam parados ou só giram: `signal` errado para `TEAM=yellow`, ou stride da feature trocou.
- `load_state_dict` com missing keys: nome de camada divergiu.
- Política joga "pior" que o checkpoint conhecido: feature normalizada com escala diferente entre treino e deploy.

Quando em dúvida, delegue ao subagente `sim2real` para uma análise focada.
