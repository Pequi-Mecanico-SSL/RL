---
description: "Rodar um checkpoint treinado de volumes/dgx_checkpoints contra o grSim usando o Docker Compose. Pergunta qual checkpoint usar se não informado e valida antes de subir."
name: "Run Checkpoint on grSim"
argument-hint: "<caminho-do-checkpoint-ou-vazio>"
agent: "grsim-deploy"
model: "Claude Opus 4.7 (copilot)"
---

Quero rodar uma policy treinada contra o grSim. Caminho do checkpoint (se fornecido): `${input:checkpoint}`.

Passos:

1. Se o caminho não foi fornecido, liste os checkpoints disponíveis com:
   ```bash
   find volumes/dgx_checkpoints -maxdepth 3 -name 'checkpoint_*' -type d | sort
   ```
   e peça ao usuário para escolher.
2. Valide que `<checkpoint>/policies/policy_blue/policy_state.pkl` existe.
3. Suba a pipeline com:
   ```bash
   CHECKPOINT_PATH="<caminho-absoluto-ou-relativo-validado>" ./start_policy.sh
   ```
4. Mostre como acompanhar os logs em paralelo:
   - `tail -f logs/policy.log`
   - `docker compose -f docker-compose.grsim.yml logs -f rl-policy`
5. Mostre o comando de parada: `./stop_policy.sh`.

Se algo falhar (checkpoint quebrado, compose v1, mismatch de obs_size), aplique o diagnóstico padrão descrito no agente `grsim-deploy`. Não tente "consertar" alterando código sem confirmação do usuário.
