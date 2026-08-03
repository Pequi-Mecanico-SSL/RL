# Campanha de melhoria da policy PPO 3v3

Registro metodologico e cumulativo. Cada fase deve declarar hipotese, mudanca unica, gate, resultado e decisao antes da proxima alteracao.

## Manifesto baseline

- Inicio: 2026-08-03.
- Branch: `experiment/policy-improvement`.
- Repo pai: `e945e9a` (`fix treino`).
- rSoccer: `c684c2bcb308c88cf8589749137991ed921daff5`.
- Artefato: `PPO_Soccer_baseline_2025-03-16/checkpoint_000003`.
- RLlib: 2.10.0; checkpoint format 1.1; Python 3.10.
- Policy: PPO self-play 3v3, `policy_blue` treinavel e `policy_yellow` defasada.
- Observacao: 77 features por frame, stack 8, total 616.
- Acao: quatro dimensoes Beta em `[-1, 1]`.
- Campo: Division B, 9 x 6 m; 30 FPS; episodio de 40 s/1200 passos.
- Modelo: MLP 300/200/100, value branch separada.
- Baseline treinado: 8.782.560 passos, 228 iteracoes, 24.942 episodios (~5,5% de 160M).
- Checkpoint disponivel mais recente: `checkpoint_000003`, associado a aproximadamente iteracao 200.
- Host: 16 CPUs, 15 GiB RAM, RTX 3060 12 GiB; nenhum processo CUDA de compute no inicio.
- Espaco livre: 83 GiB no inicio e 33 GiB apos os builds; verificar antes de cada rodada.

## Evidencia anterior

Cross-play fiel ao env historico, 20 episodios por confronto contra yellow fixa no ckpt0:

| Blue | Gols blue | Gols yellow | Timeouts |
|---|---:|---:|---:|
| ckpt0 | 1 | 1 | 18 |
| ckpt1 | 6 | 1 | 13 |
| ckpt2 | 12 | 0 | 8 |
| ckpt3 | 14 | 0 | 6 |

Conclusao: houve progressao real; avaliacao em espelho subestimava a policy. O nivel absoluto ainda e baixo e o treino terminou cedo.

## Regras da campanha

1. Nao alterar reward, observacao e algoritmo simultaneamente.
2. Validar restore e um smoke curto antes de treino prolongado.
3. Avaliar candidatos contra pool fixo e espelho; reward bruto nao seleciona sozinho.
4. Preservar checkpoints e configuracoes de cada rodada.
5. Replanejar apos cada gate; falha interrompe a progressao automatica.
6. Nao executar em robo real.

## Plano inicial

### Fase A - diagnostico sem treino

Hipotese A1: a distribuicao Beta continua larga; `mean` e fraca apesar de `sample` marcar gols.

Gate: medir alpha, beta, media, desvio e saturacao por acao em trajetorias do ckpt3.

### Fase B - reproducibilidade do restore

Hipotese B1: o ckpt3 restaura integralmente no stack historico RLlib 2.10.0.

Gate: container historico carrega o Algorithm, executa uma iteracao curta controlada e salva checkpoint sem NaN/Inf.

### Fase C - continuacao conservadora

Mudanca unica proposta: continuar o baseline sem mudar reward/observacao/modelo, com checkpoints frequentes e limite curto inicial.

Gate: cross-play do candidato contra ckpt0/ckpt2/ckpt3 e espelho. Promover somente se melhorar saldo e reduzir timeout sem regressao numerica.

## Diario

### 2026-08-03 - Preparacao

- Worktree isolado criado em `/home/marcos/Documentos/RL-policy-improvement`.
- `main`, branch `grsim` e containers de deploy nao foram alterados.
- Divergencia critica registrada: o `main` atual usa rSoccer v1.2.0; o baseline exige `c684c2b`.
- Proxima decisao: medir a Beta e validar restore antes de consumir uma rodada longa de treino.

### 2026-08-03 - Fase A: distribuicao Beta

Comando: `scripts/analyze_beta_trajectories.py` no env historico `c684c2b`,
checkpoint 000003, 30 episodios, seeds 1000..1029, CPU-only.

Resultado salvo em `experiment_results/beta_ckpt3_30ep.json`:

- 164.652 agent-steps.
- Terminais: 10 gols blue, 3 yellow, 17 timeouts.
- Desvio padrao medio por acao `[x, y, omega, kick]`: `[0,420, 0,417, 0,482, 0,498]`.
- Amostras com `abs(action) > 0,8`: `[20,8%, 18,7%, 13,9%, 12,9%]`.
- Kick positivo em 55,0% dos agent-steps.
- Alpha mediano: `[1,379, 1,877, 1,441, 1,524]`.
- Beta mediano: `[1,647, 1,443, 1,311, 1,313]`.

Decisao: H3 confirmada. A policy depende de uma distribuicao larga, sobretudo
em `omega` e `kick`; nao promover `mean` como modo nominal. A primeira rodada
de continuacao preservara reward, observacao e modelo, mas testara reducao
gradual de entropia somente depois de um restore inalterado passar.

### 2026-08-03 - Fase B: preparacao do restore

- Estado do ckpt3: iteracao 200, 7.704.000 env steps e 46.224.000 agent steps.
- Proximo stop de smoke: 7.742.520 env steps (uma iteracao adicional).
- `RL_train.py` recebeu apenas parametros operacionais: config, restore, stop,
	diretorio de saida e nome do experimento; os defaults historicos permanecem.
- Imagem historica em construcao como `rl-policy-training:c684c2b`.
- Gate pendente: CUDA disponivel, imports completos, restore, uma iteracao e
	checkpoint final sem NaN/Inf.

#### Falha de build e replanejamento

O primeiro build falhou ao instalar `rSim` do `main`: CMake 4.4 recusou o
`cmake_minimum_required` do pybind11 legado. O Dockerfile tambem deixava Torch e
rSim sem versao fixa, logo uma rebuild nao era reproduzivel.

Evidencias:

- `rSim` HEAD observado: `c30ec84bc07de9ae60c6d5a1d5e9283832b46d9a`.
- Runtime standalone validado: Ray 2.10.0, Torch 2.0.1, `rc-robosim` 1.2,
	`rsoccer-gym` 1.0.0.
- Erro: pybind11 antigo rejeitado por CMake 4 sem policy minima.

Correcao operacional, sem alterar treino/fisica:

- base fixada em `python:3.10-slim-bookworm`;
- Torch fixado em `2.0.1+cu118`;
- rSim fixado em `c30ec84bc07de9ae60c6d5a1d5e9283832b46d9a`;
- `CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5` apenas para compatibilidade
	de build.

Novo build iniciado com log em `/tmp/rl-policy-training-build.log`.

#### Gate de runtime apos build

- Imagem: `sha256:53eec0c5ff3783d2c5440a2f6262eb1dd1d7472fa27df6f14e24499fa8ede059`.
- CUDA passou: Torch 2.0.1+cu118 reconheceu RTX 3060.
- Versoes: Ray 2.10.0, Gymnasium 1.3.0, rc-robosim 1.2,
	rsoccer-gym 1.0.0.
- Import do treino falhou antes de iniciar Ray: `pyarrow 25.0.0` removeu
	`PyExtensionType`, ainda usado pelo Ray 2.10.0.
- Decisao: fixar `pyarrow==20.0.0`, versao ja validada na imagem standalone,
	e repetir o gate. Nenhum treino foi iniciado.
- Espaco livre apos build: 35 GiB; monitorar antes de treinos prolongados.

### 2026-08-03 - Fase B: restore e iteracao 201 concluidos

- Imagem final: `rl-policy-training:c684c2b`, digest
	`sha256:3a94c6cf43701204cc15c8834619b1d3d0deaef99a506857e21e16af2cc7e1c9`.
- Runtime: Ray 2.10.0, Torch 2.0.1+cu118, CUDA ativa na RTX 3060 e
	`pyarrow==20.0.0`.
- Restore do ckpt3 executou a iteracao 201 ate 7.742.520 env steps, 36
	episodios, sem NaN/Inf, traceback ou erro.
- Metricas da iteracao: reward medio `-875,308`, comprimento medio `896,417`,
	`score=0,13`, entropy media `-0,839` e 76,2 s de treino.
- O RLlib 1.1 preservou `/root/ray_results` do estado restaurado. A persistencia
	correta exige montar o diretorio host diretamente nesse caminho.
- Checkpoint completo validado: cada policy possui 16 tensores e 531.709
	parametros finitos. Blue tem L2 `60,9871`; yellow tem L2 `60,9397`.
- O checkpoint completo inclui objetos cloudpickle/optimizer e nao e portatil
	para o runtime CPU de inferencia. `scripts/export_inference_checkpoint.py`
	gera um artefato separado apenas com pesos NumPy e metadados; o export iter201
	foi carregado com sucesso no runtime `rl-grsim-rl-policy`.

Gate esportivo de continuidade, mesmos 20 seeds da referencia, blue iter201
contra yellow ckpt0:

| Blue | Gols blue | Gols yellow | Timeouts | Steps medios |
|---|---:|---:|---:|---:|
| ckpt3 | 14 | 0 | 6 | nao registrado |
| iter201 | 16 | 0 | 4 | 875,4 |

Returns medios do iter201: blue `-59,29`, yellow `-258,70`.

Decisao: B1 confirmada e nenhuma regressao imediata observada. A diferenca de
dois episodios e pequena para declarar melhoria. A Fase C comecara por um braco
controle de 25 iteracoes sem alterar hiperparametros, reward, observacao ou
modelo. O objetivo e medir a tendencia natural da continuacao antes de testar
um schedule de entropia. Interromper se houver NaN/Inf, erro, falta de espaco ou
checkpoint invalido.
