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

### 2026-08-03 - Fase C: primeira tentativa do controle interrompida

- Alvo: iter201 ate iter226, 8.705.520 env steps, sem mudanca de configuracao.
- A tentativa usou limite Docker de 13 GiB e falhou antes da primeira iteracao:
	o actor PPO morreu por `SIGKILL`/fim de conexao apos 83 s.
- O Ray indicou OOM como causa provavel. Nao houve OOM global no journal do
	kernel, NaN/Inf, resultado ou checkpoint; `result.json` ficou vazio.
- O host possui 15 GiB e nenhum swap. Apos a falha havia 7,5 GiB disponiveis e
	nenhum outro container/GPU workload relevante.

Decisao: nenhum resultado de treino foi aceito. A hipotese operacional e que o
limite cgroup de 13 GiB matou o actor, nao que a policy divergiu. Antes de repetir
o controle, executar somente uma iteracao sem `--memory`. Se ela falhar, reduzir
workers/envs em configuracao operacional separada; se passar, retomar o controle
sem limite cgroup e manter os mesmos hiperparametros PPO.

#### Smoke de recuperacao

- Repeticao sem `--memory` concluiu a iteracao 202 em 76,4 s e 7.781.040 env
	steps, com 35 episodios, reward medio `-916,565` e comprimento medio `922`.
- Gate estrito do log: sem NaN/Inf, `Traceback` ou nivel `ERROR`.
- Checkpoint completo valido: 16 tensores e 531.709 parametros finitos por
	policy; L2 blue `61,0170` e yellow `60,9397`.

Decisao: a hipotese de OOM pelo limite cgroup foi confirmada pelo teste
discriminante. O checkpoint iter202 nao sera usado no controle. A rodada de 25
iteracoes sera reiniciada do iter201, sem limite Docker de memoria, mantendo a
mesma origem e os mesmos hiperparametros planejados.

#### Segunda tentativa: OOM do Ray confirmado

- Sem limite cgroup, a rodada concluiu somente as iteracoes 202..206 e falhou
	antes da 207. Nao houve checkpoint porque a frequencia historica era 50.
- O raylet registrou explicitamente workers mortos por pressao de memoria: um
	worker no primeiro evento e sete no evento acumulado seguinte.
- Apos perder workers, a coleta terminou com batch GAE inconsistente:
	`rewards (41,)` contra `vpred (40,)`. Esse erro e consequencia da degradacao,
	nao evidencia de divergencia da policy.
- As cinco iteracoes validas chegaram a 7.999.320 env steps, `score=0,19`, sem
	atualizacao da yellow. Esses pesos nao foram salvos e nao serao usados.

Decisao: reduzir somente o paralelismo de coleta em
`config.control-lowmem.yaml`: 3 workers em vez de 6, mantendo 2 envs por worker,
batch PPO de 38.520 e todos os hiperparametros de aprendizado. Checkpoints passam
a cada 5 iteracoes para limitar perda operacional. Executar primeiro 3 iteracoes
desde iter201; somente sem morte de worker, NaN/Inf ou erro retomar ate iter226.

#### Smoke low-memory aprovado

- Iteracoes 202..204 concluidas em 354,8 s, ate 7.858.080 env steps.
- Gate do log: nenhum OOM, worker morto, NaN/Inf, `Traceback` ou nivel `ERROR`.
- Checkpoint iter204 valido: 16 tensores e 531.709 parametros finitos por policy;
	L2 blue `61,1225` e yellow `60,9397`.
- A comparacao estrutural com `config.yaml` confirmou somente tres diferencas:
	`num_workers 6 -> 3`, `num_cpus 7 -> 4` e `checkpoint_freq 50 -> 5`.

Decisao: continuar a mesma trajetoria valida do iter204 ate iter226. Esse smoke
faz parte do braco controle porque partiu do iter201 e nao alterou nenhum
hiperparametro PPO, reward, observacao ou modelo.

#### Continuacao low-memory reprovada e artefatos em quarentena

O `policy-verifier` auditou `config.control-lowmem.yaml`, o log, `progress.csv`
e os contadores internos dos checkpoints. Veredito independente: **REPROVADO**.

Evidencias:

- Iteracoes 205..210 avancaram exatamente 38.520 env steps cada, de 7.896.600
	ate 8.089.200, com 36..41 episodios por iteracao.
- O primeiro evento invalido ocorreu depois da iter210: dois actors morreram as
	18:50:30 e o raylet confirmou seis workers mortos por pressao de memoria as
	18:50:52.
- A iter211 avancou somente 12.840 env steps, um terco do batch planejado, e
	registrou zero episodios. Varias iteracoes seguintes mantiveram zero episodios
	e steps congelados apesar de o numero de iteracao continuar aumentando.
- O processo encerrou sem container ativo. O ultimo registro persistido chegou
	a iter247 e 8.217.600 env steps, abaixo do alvo de 8.705.520.
- `checkpoint_000001` corresponde a iter210: 8.089.200 env steps treinados e
	amostrados. A validacao encontrou 16 tensores e 531.709 parametros finitos por
	policy; L2 blue `61,4541` e yellow `60,9397`.
- O host permaneceu sem swap. Ao fim havia 6,4 GiB de memoria disponivel e 33
	GiB livres em disco; imagens e cache Docker ocupam espaco relevante, mas nao
	foram removidos automaticamente para preservar outros workloads.

Decisao: iter210 e o ultimo checkpoint elegivel para restore. Todos os resultados
da iter211 em diante e `checkpoint_000002`..`checkpoint_000008` ficam em
quarentena: nao avaliar, exportar ou promover. Antes de nova rodada, implementar
um watchdog que encerre no primeiro worker morto/OOM, batch diferente de 38.520,
zero episodios ou steps congelados; submeter config e comando ao
`policy-verifier` em preflight.

#### Watchdog implementado; smoke 2-worker bloqueado por recursos

- `RL_train.py` agora desabilita recriacao de workers e retries do Tune, valida
	batch, iteracao, counters cumulativos, episodios, workers/restarts e finitude
	dos pesos antes de atualizar a yellow ou permitir checkpoint.
- Cinco testes no runtime fixo passaram: iter210 aceita; iter211, chave ausente,
	counters resetados e peso NaN rejeitados.
- O validador completo do iter210 foi persistido em
	`experiment_results/iter210_checkpoint_validation.json`: iteracao 210, env
	steps 8.089.200, agent steps 48.535.200 e optimizer blue finito.
- Novo braco operacional preparado em `config.control-2w.yaml`. A comparacao
	estrutural com a config 3-worker mostrou apenas `num_cpus 4 -> 3` e
	`num_workers 3 -> 2`; PPO, reward, observacao e modelo permanecem iguais.
- O `policy-verifier` aprovou os gates com ressalvas, mas reprovou a execucao
	atual: outro workload consumia 2,663 GiB RAM, 3.388 MiB VRAM, 105% CPU e 58%
	de GPU. Havia somente 6,93 GiB de RAM disponivel e nenhum swap.

Decisao: nao iniciar nem parar o workload alheio. O smoke permanece bloqueado
ate duas medicoes independentes confirmarem RAM disponivel >= 10 GiB, GPU livre
>= 10.240 MiB, utilizacao GPU <= 10%, nenhum processo CUDA concorrente e disco
livre >= 30 GiB. Quando liberado, restaurar somente iter210, usar stop exato
8.127.720 e aceitar exclusivamente a iter211 com 38.520 env steps novos, 2
workers saudaveis, zero restarts/faulty episodes e episodios positivos.

#### Tentativa de execucao bloqueada no preflight

- Medicao em 2026-08-03T17:31:51-03:00: 9.555.988.480 bytes de RAM disponivel,
	nenhum swap, GPU com 11.496 MiB livres e 40% de utilizacao, sem processo CUDA
	listado, e 35.210.395.648 bytes livres em disco.
- Containers concorrentes: somente `jurisprudencia-postgres`, com 22,56 MiB de
	RAM e 0% CPU.
- RAM disponivel ficou abaixo de 10 GiB e utilizacao da GPU acima de 10%; os
	demais gates passaram.

Decisao: preflight **REPROVADO**. O smoke nao foi iniciado e nenhum workload foi
interrompido. Repetir as duas medicoes quando RAM e GPU estiverem naturalmente
dentro dos limites.

### 2026-08-03 — Consolidacao na branch grsim

O diretorio principal /home/marcos/Documentos/RL foi movido para a branch local
`grsim`, criada a partir de 139a7f1 (fix/grsim-policy-validation, 1 commit a
frente de origin/grsim=0e5f906). A campanha experiment/policy-improvement
(0f4891f) foi mesclada nesta branch. Base do merge: e945e9a.

Resolucao dos 4 conflitos:
- Dockerfile: versao da campanha (torch==2.0.1+cu118 e rSim pinado em
  c30ec84; reproduzivel). O deploy grSim usa Dockerfile.policy, nao este.
- RL_train.py: versao da campanha (contrato historico e945e9a + watchdog
  fail-fast validate_train_result/validate_policy_weights).
- requirements.txt: versao grsim (superset; ja contem pyarrow==20.0.0,
  dill, packaging<23, rc-robosim).
- .gitignore: mescla manual (volumes/*, logs, grSim/, training_runs/).

Fatos de contrato verificados no merge:
- rewards.py e observations.py ficaram na versao refatorada do lado grsim
  (compativel com rSoccer v1.2.0, submodulo pinado em 2520207). As assinaturas
  mudaram de Field/Frame para dict — INCOMPATIVEL com o env historico c684c2b.
- CONSEQUENCIA: continuar treino do iter210 exige montar as versoes
  historicas por cima do workdir no container, por exemplo:
    git show e945e9a:rewards.py > /tmp/rewards_hist.py
    docker run ... -v /tmp/rewards_hist.py:/campaign/rewards.py:ro ...
  O RL_train.py da campanha importa DENSE_REWARDS/SPARSE_REWARDS, presentes
  em ambas as versoes, mas a semantica difere; o smoke iter211 deve usar a
  versao historica.
- config.yaml ficou na versao grsim (checkpoint_freq 25, stack_size etc.);
  o treino da campanha usa config.control-*.yaml proprios, sem impacto.
