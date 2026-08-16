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
  historicas de AMBOS os contratos por cima do workdir no container:
    git show e945e9a:rewards.py > /tmp/rewards_hist.py
    git show e945e9a:observations.py > /tmp/observations_hist.py
    docker run ... \
      -v /tmp/rewards_hist.py:/campaign/rewards.py:ro \
      -v /tmp/observations_hist.py:/campaign/observations.py:ro ...
  O RL_train.py da campanha importa DENSE_REWARDS/SPARSE_REWARDS, presentes
  em ambas as versoes, mas a semantica difere; o smoke iter211 deve usar a
  versao historica.
- config.yaml ficou na versao grsim (checkpoint_freq 25, stack_size etc.);
  o treino da campanha usa config.control-*.yaml proprios, sem impacto.

#### Evidencias pos-merge (auditoria policy-verifier)

- git status limpo; diff HEAD vs experiment/policy-improvement em RL_train.py e
  Dockerfile vazio; diff HEAD vs 139a7f1 em requirements.txt, rewards.py,
  observations.py, config.yaml, scripts/model/, deploy_policy_grsim.py,
  docker-compose.grsim*.yml, Dockerfile.policy, tests/test_grsim_contract.py,
  *_pb2.py e scripts/sim2real/ vazio.
- gitlink rSoccer = 2520207 (v1.2.0), coerente com o deploy grSim validado.
- Checkpoint iter210 copiado para training_runs/ deste diretorio: SHA-256
  identico em 8/8 arquivos (manifesto em
  experiment_results/iter210_checkpoint_sha256.txt).
- 5/5 testes do watchdog e validacao estrutural do checkpoint reproduzidos no
  container a partir deste diretorio (-w /campaign).
- deploy_policy.py corrigido: load_state_dict agora usa strict=True (o
  entrypoint validado deploy_policy_grsim.py ja usava strict=True).

---

# Campanha 2 — melhoria metodologica da policy (2026-08-15)

Autorizacoes do usuario: GPU do host liberada para treino; trabalho autonomo;
nao rodar treinos inteiros — apenas o necessario para validar cada ideia.

## Protocolo desta campanha

1. **Uma variavel por braco.** Cada experimento muda exatamente uma coisa em
   relacao ao braco de controle e declara hipotese, predicao, custo, metrica
   primaria e criterios de aceite E de rejeicao ANTES de rodar.
2. **Debate previo obrigatorio.** Toda hipotese passa pelo subagente
   `idea-debater` (segunda opiniao adversarial, mesmo modelo); veredito
   registrado aqui. `policy-verifier` audita plano e resultados.
3. **Metrica primaria: cross-play.** 20 episodios do candidato (blue) contra
   pool fixo de yellow (ckpt0 e ckpt3 do baseline) + espelho. Reportar gols
   pro/contra e timeouts. Reward bruto nao seleciona sozinho.
4. **Probes curtos.** Bracos de treino usam 25 iteracoes (~963.000 env steps,
   1 checkpoint em freq 25) antes de qualquer extensao. Extensao so com
   melhora na metrica primaria.
5. **Contrato fixo.** Observacao (77x8), acao (4 Beta), arquitetura
   (300/200/100) e rSoccer@c684c2b nao mudam nesta campanha; mudancas de
   contrato reabrem gates de paridade grSim e ficam para campanha propria.
   Treino sempre monta rewards.py e observations.py de e945e9a (ver
   consolidacao acima).
6. **Gates operacionais.** Preflight de recursos (2 medicoes), watchdog
   fail-fast ativo, restore exclusivo de checkpoint validado, quarentena
   imediata em OOM/worker morto.

## Backlog de hipoteses (a debater)

- **H0 (infra, pre-requisito):** o pipeline fail-fast reproduz 1 iteracao
  limpa a partir do iter210 (smoke iter211, aceite exato ja definido).
- **H1 (subtreino):** o baseline parou cedo (~5,5% do orcamento); continuar
  treino conservador melhora cross-play monotonicamente nos proximos
  25-50 iteracoes. Predicao: saldo de gols sobe e timeouts caem vs iter210.
- **H2 (inferencia, custo zero de treino):** com alphas medios ~1,6-2,3
  (beta_ckpt3_30ep.json), a Beta e larga; `deterministic` (mean) pode ser
  pior que `sample` no deploy. Escolher modo por avaliacao offline.
- **H3 (entropy_coeff):** reduzir entropia afia a Beta tarde no treino e
  converte posse em gol. Probe de 25 iteracoes vs controle H1.
- **H4 (lr schedule):** decaimento de lr estabiliza a continuacao. So se H1
  mostrar instabilidade.
- **H5 (pesos de reward):** rebalancear 0.7/0.1/0.1/0.1 para reforcar
  ataque. Mais arriscado (muda shaping, nao muda contrato de deploy);
  somente apos H1/H3 esgotarem.

## Diario da campanha 2

### 2026-08-15 — Debate do backlog (idea-debater, GPT-5.6 Sol)

Fatos novos levantados pelo debatedor (auditáveis no baseline):
- entropia Beta caiu de -0,31 (iter1) para -1,04 (iter228) com entropy_coeff
  0.01 — a policy JA se concentra; premissa de H3 enfraquecida.
- custo real ~166-174 s/iteracao (6 workers); 25 iteracoes ~1-2 GPU-h.
- score do self-play estava 0,99 na iter228: o callback pode estar copiando
  blue para yellow quase toda iteracao (self-play com defasagem ~1 iteracao).

Vereditos:
- H0 smoke iter211: APOIO COM MUDANCAS — aprova pipeline, nao estabilidade;
  H1 ganha gate operacional extra apos 3 iteracoes.
- H1 continuacao 25 iter: APOIO COM MUDANCAS — avaliacao primaria pre-definida
  so na iter235; primaria = taxa de vitoria vs ckpt3 (ckpt0 = gate de nao
  regressao; espelho = diagnostico); 20 eps para triagem, 80 eps pareados se
  promissor; aceite = IC95% da diferenca pareada > 0.
- H2 mean vs sample offline: APOIO COM MUDANCAS — so mean vs sample (sem
  temperatura), yellow fixa em sample, 80 seeds iguais vs ckpt3; empate
  mantem mean. Nao rodar simultaneo a treino (compete por CPU/RAM).
- H3 entropy 0.003 fixo: APOIO COM MUDANCAS — somente apos H1; sem schedule.
- H4 lr: CONTRA no momento (sem instabilidade demonstrada; reabrir com
  regressao em 2 checkpoints + losses/grad_gnorm).
- H5 reward: CONTRA nesta campanha — antes, decomposicao offline por
  componente vs gols futuros.
- NOVA Hsync-audit (custo ~zero): auditar frequencia real de sync da yellow
  no baseline antes de considerar opponent pool.

Sequencia adotada: H0 → auditorias CPU (sync, baseline iter210 cross-play,
decomposicao reward) → H2 offline → H1 (25 iter) → [H3 se H1 nao resolver
conversao] → gate curto grSim do vencedor. Orcamento reservado: ate 7 GPU-h.

### 2026-08-15 — Hsync-audit (custo zero) — CONCLUIDO

Auditoria do progress.csv do baseline (228 iteracoes):
- 162 syncs blue→yellow (score>0.6) em 228 iteracoes; primeiro sync na iter58.
- A partir da iter58, gap entre syncs: media 1,06, mediana 1, maximo 2.
- Episodios por iteracao: media 109 (min 24, max 182) — o deque de 100
  episodios renova quase inteiro a cada iteracao, entao o gate 0.6 fica
  permanentemente aberto.

Conclusao: o self-play degenerou em adversario quase-espelho (yellow = blue
de ~1 iteracao atras) desde cedo. Implicacoes: (a) diversidade de adversario
minima — candidato natural a hipotese futura de opponent pool/threshold mais
alto; (b) reforca usar pool FIXO (ckpt0/ckpt3) como metrica, nunca o espelho.

### 2026-08-15 — Baseline cross-play do iter210 + triagem H2

Harness: scripts/evaluate_checkpoints_cpu.py no container historico, contrato
montado (rewards.py e945e9a sha256 9379220616..., rSoccer c684c2b via mount
read-only do worktree; observations.py NAO existe em e945e9a — o env legado
calcula as 77 obs internamente, correcao ao registro da consolidacao).

Triagem 20 seeds, ambos os times no MESMO modo (protocolo antigo):
| confronto | blue | gols | W/L/T |
|---|---|---|---|
| iter210 vs ckpt3 | det | 0x20 | 0/20/0 |
| iter210 vs ckpt3 | sample | 4x1 | 4/1/15 |
| iter210 vs ckpt0 | det | 0x0 | 0/0/20 |
| iter210 vs ckpt0 | sample | 12x0 | 12/0/8 |

CONFUSOR IDENTIFICADO: em det x det o env e deterministico → 20 episodios sao
1 episodio repetido; o 0x20 vs ckpt3 e uma unica trajetoria perdedora. Nao
usar det x det como metrica.

Matriz H2 corrigida (harness ganhou --yellow-mode; yellow FIXA em sample,
mesmos 20 seeds):
| blue | gols | W/L/T |
|---|---|---|
| deterministic | 10x0 | 10/0/10 |
| sample | 4x1 | 4/1/15 |

Sinal inverte a intuicao inicial: blue deterministic > blue sample contra
yellow sample. Extensao para 80 seeds pareados em andamento (aceite: IC95%
da diferenca pareada). Nota: iter210 (10 iteracoes alem do ckpt3) vence
ckpt3 por 10x0 em modo det — evidencia preliminar pro-H1.

### 2026-08-15 — H2 CONCLUIDO: modo de inferencia (mean vs sample)

80 seeds pareados, iter210 (blue) vs ckpt3 (yellow FIXA em sample):
| blue | gols | W/L/T |
|---|---|---|
| deterministic | 35x1 | 35/1/44 |
| sample | 16x7 | 16/7/57 |

Diferenca pareada por seed (win=+1, timeout=0, loss=-1), det - sample:
media +0,312, IC95% [+0,162, +0,463] — exclui zero com folga.

DECISAO H2: modo `deterministic` (mean) e superior para inferencia/deploy
desta policy e passa a ser o modo padrao de avaliacao do candidato (yellow
permanece sample para nao degenerar o env deterministico). Recomendacao para
o deploy grSim: migrar de `sample` para `deterministic` — isso reabre o gate
de ativacao gradual do grSim (mudanca de modo), a validar em rodada propria.
Confusores tratados: mesmos 80 seeds nos dois bracos, yellow fixa em modo
unico, comparacao pareada.

### 2026-08-15 — Debate: relaxamento do preflight (idea-debater)

Veredito APOIO COM MUDANCAS para executar H0 agora:
- gate `GPU utilization <= 10%` REMOVIDO (media atividade grafica do desktop,
  nao concorrencia CUDA); substituido por: zero compute apps + VRAM livre
  >= 10.240 MiB, utilizacao registrada como telemetria;
- gate de RAM corrigido para bytes: aceite de H0 com MemAvailable
  >= 9.500.000.000 bytes (relaxamento EXCLUSIVO do smoke; nao chamar de 10 GiB);
- container de treino com --memory=7g --memory-swap=7g (protege o host de
  verdade; cap de 12g nao protegeria com ~9 GiB disponiveis);
- monitor externo amostrando MemAvailable, memoria do container e VRAM;
- aceite H0 inalterado + pico do container < 6,5 GiB + zero eventos OOM;
- H0 NAO autoriza H1: probe intermediario de 8 iteracoes obrigatorio antes
  das 25 (o OOM anterior surgiu apos 6 iteracoes validas), aceite = pico de
  memoria estavel nas ultimas 4 iteracoes com folga >= 0,5 GiB do cap.

### 2026-08-15 — H0 tentativa 1 (2 workers, cap 7g): REPROVADO por OOM

Ray memory monitor matou RolloutWorker no startup: 6,88/7,00 GB no cgroup
(PPO.train 2,26 GB + workers 1,34/1,12 GB + driver 0,32 GB + object store em
/dev/shm dentro do cap). fail_fast funcionou: TuneError, zero iteracoes
aceitas, nenhum checkpoint gerado, host protegido pelo cap.
Decisao pre-registrada do debate: NAO subir o cap; reduzir para 1 worker.
Nova config config.control-1w.yaml (unico diff da 2w: num_cpus 3→2,
num_workers 2→1). Repetindo H0 com cap 7g.

### 2026-08-15 — H0 tentativa 2 (1 worker, cap 7g, shm 4g): REPROVADO por OOM

Mesmo padrao: Ray memory monitor matou worker. Diagnostico: o object store
(Plasma) e alocado em /dev/shm com ~30% da memoria do no detectado e CONTA no
cgroup: processos (~4,4 GB: PPO.train 2,26 + worker 1,3 + raylet/gcs/driver
~0,8) + plasma ~2,1 GB ≈ 93% do cap → monitor mata em 95%.
Correcao (mantendo cap 7g do debate): reduzir --shm-size para 1536m, que
limita o object store por construcao. Batch trafegado por iteracao ~0,6 GB,
cabe com folga. Tentativa 3 = 1 worker, cap 7g, shm 1536m.

### 2026-08-15 — H0 APROVADO (tentativa 4: 1 worker, cap 7g, shm 1536m)

Tentativa 3 (shm 1536m) TERMINATED com iter211 exata em 328 s, mas o
checkpoint foi para /root/ray_results DENTRO do container efemero:
tune.run(local_dir=...) e ignorado no Ray 2.10. Correcao sem tocar codigo:
montar /root/ray_results no host. Tentativa 4 = mesma config com o mount.

Aceite verificado (PPO_Soccer_b6960, training_runs/h0_smoke/ray_results/
control_1w_smoke_iter211/):
- training_iteration=211; ts=8.127.720; this_iter=38.520;
  agent=48.766.320 (=6x env); episodes_this_iter=38>0;
- 1 worker saudavel; 0 restarts; 0 faulty episodes; EXIT=0;
- checkpoint_000000 integro: counters exatos, blue l2=61,537 finito,
  optimizer 32 tensores; validado no container read-only;
- pico do container na janela do run: 4,29 GiB < 6,5 GiB; zero eventos OOM;
- custo: ~320 s/iteracao com 1 worker.

Receita operacional canonica de treino (registrada): cap 7g + swap 7g,
shm 1536m, 1 worker (config.control-1w.yaml), mounts do contrato historico,
mount obrigatorio de /root/ray_results para persistencia.

H1 desenho (conforme debate): run A = restore iter210, stop 8.397.360
(8 iteracoes, ate iter218); gate de memoria estavel; run B = retomar do
checkpoint_at_end ate 9.052.200 (iter235). Avaliacao primaria SO na iter235:
blue deterministic vs ckpt3 yellow sample, 20 seeds triagem → 80 pareados.

### 2026-08-15 — Condicoes do policy-verifier cumpridas (pre-run A)

Parecer do verificador sobre H0: INCONCLUSIVO (faltavam artefatos persistidos);
plano H1: REPROVADO com correcoes baratas. Todas atendidas:
1. Validacao H0 persistida: experiment_results/h0_iter211_checkpoint_validation.json
   (iter211, counters exatos, 16 tensores/531.709 params por policy,
   optimizer blue 32 tensores finitos).
2. Delta tensor-a-tensor iter210→211 persistido em
   experiment_results/h0_iter211_weight_delta.json: blue l2_delta=2,688
   (rel 4,37%, max_abs 0,0552 na value branch, 531.709/531.709 elementos
   mudaram, finito); yellow delta=0 exato (score 0,1 nao disparou sync,
   como esperado). Update real e sadio → H0 ACEITO.
3. Manifesto de ambiente: experiment_results/h0_environment_manifest.txt
   (rewards sha256 9379...7ced, imagem sha256:3a94c6cf..., rSoccer c684c2b).
4. Baseline ckpt0 no protocolo final (blue det, yellow sample, 80 seeds
   0..79): 46W/0L/34T, gols 46x0
   (experiment_results/crossplay_iter210_vs_ckpt0_bdet_ysample.jsonl).
5. Gate de memoria quantitativo para run A: pico de cada uma das 4 ultimas
   iteracoes <= 6,5 GiB, amplitude <= 0,5 GiB, tendencia <= 0,1 GiB/iteracao.
6. Sem optional stopping: avaliacao final SEMPRE com os 80 seeds fixos 0..79.
7. Gate ckpt0: diferenca pareada candidato−iter210 com IC95% LB >= 0.
8. Conclusao restrita a "iter235 supera iter210 neste protocolo" (1 training
   seed; sem alegacao de monotonicidade).

### 2026-08-15 — Run A BLOQUEADO por preflight (veredito CONTRA do debate)

Preflight do run A: MemAvailable 7,93/7,88 GB (< 9,5 GB). Debate concluiu
CONTRA rodar agora: na janela do H0 o MemAvailable caiu de 9,35 para 2,95 GB
com o container em apenas 4,29 GiB; partir de 7,9 GB deixaria o host ~1,4 GB.
Aborto em 1,5 GB reage tarde (queda de 2,76 GiB entre amostras de 7 s) e
reduzir o cap contaminaria H1 (mudaria o envelope operacional vs smoke).

Condicoes de liberacao do run A (registradas para execucao):
- MemAvailable >= 9.500.000.000 bytes em 2 medicoes separadas por 20 s;
- zero compute apps CUDA; VRAM livre >= 10.240 MiB; disco >= 30 GB;
- monitor externo com amostragem de 1 s;
- abort se MemAvailable < 2.500.000.000 bytes OU container >= 6,5 GiB.

Comando pronto (run A, 8 iteracoes ate iter218):
  docker run --rm --gpus all --memory=7g --memory-swap=7g --shm-size=1536m \
    -w /campaign -v $PWD:/campaign \
    -v $PWD/training_runs/h1_runA/ray_results:/root/ray_results \
    -v /tmp/contrato_hist/rewards.py:/campaign/rewards.py:ro \
    -v /home/marcos/Documentos/RL-policy-improvement/rSoccer:/campaign/rSoccer:ro \
    -e PYTHONPATH=/campaign:/campaign/rSoccer rl-policy-training:c684c2b \
    python RL_train.py --config config.control-1w.yaml \
    --restore /campaign/training_runs/control_lowmem_to226_ray_results/control_lowmem_to226/PPO_Soccer_286cf_00000_0_2026-08-03_18-35-53/checkpoint_000001 \
    --stop-timesteps 8397360 --experiment-name h1_runA_iter218
Depois do run A: validar checkpoint iter218 + gates de memoria; run B =
restore iter218, stop 9.052.200 (iter235); avaliacao final 80 seeds fixos.

### 2026-08-15 11:33 — Run A DESBLOQUEADO e lançado
Preflight repassou: MemAvailable 9,75/9,85 GB (2 medicoes, 20 s), 0 compute
apps CUDA, VRAM livre 11.263 MiB, disco 36,8 GB. Monitor externo de 1 s ativo
em training_runs/h1_runA/monitor.log. Run A lancado 11:34 com a receita
canonica (restore iter210, stop 8.397.360 = iter218). Checagem inicial:
container 3,7-4,3 GiB, host 4,4-4,7 GB livres — dentro do envelope do smoke.

### 2026-08-15 12:19 — Run A concluido; parecer post-result; run B lancado

Run A TERMINATED (EXIT=0): 8 iteracoes 211..218 exatas (this_iter=38.520,
episodios 35..44, 1 worker, 0 restarts/faulty), ts final 8.397.360.
Checkpoint iter218 validado e PERSISTIDO em
experiment_results/h1_iter218_checkpoint_validation.json: counters exatos
(env 8.397.360, agent 50.384.160 = 6x), blue l2 61,9916 (progrediu de
61,4541), yellow l2 60,9397 = iter210 (sem sync, esperado), optimizer 32
tensores finitos.

Gate de memoria: com janelas alinhadas as iteracoes reais (auditoria do
verificador), picos 215..218 = [6,515; 6,326; 6,380; 6,291] GiB — amplitude
0,224 e tendencia -0,062 GiB/iter PASSAM; teto de 6,5 falhou por 15 MiB na
iter215; pico global 6,778 GiB (96,8% do cap); MemAvailable do host tocou
2,04 GB sem abort automatico (falha operacional reconhecida).

policy-verifier: APROVADO COM RESSALVAS. iter218 elegivel para restore APOS
persistir validacao (feito). Run B autorizado com a MESMA receita, mas com
watchdog ATIVO obrigatorio: abort automatico se MemAvailable < 2,5 GB ou
container >= 6,5 GiB, amostragem 1 s (scripts/watchdog_host_memory.sh).
Qualquer aborto poe em quarentena apenas o trecho pos-iter218.

DGX considerada e descartada nesta janela (sobrecarregada por outros
usuarios); campanha segue no host local.

Run B lancado 12:18 (preflight: MemAvailable 10,74/10,73 GB, 0 CUDA apps,
VRAM 11.544 MiB, disco 37,2 GB): restore iter218, stop 9.052.200 (iter235),
container --name h1_runB, watchdog em training_runs/h1_runB/watchdog.log.

### 2026-08-15 12:23 — Run B tentativa 1 ABORTADA pelo watchdog (falso positivo)

O watchdog matou o container aos ~5 min: media 6,88 GiB por memory.current.
Diagnostico: memory.current do cgroup v2 inclui page cache (reciclavel pelo
kernel ate o cap), enquanto o gate de 6,5 GiB foi definido na semantica do
docker stats (usage - inactive_file), usada no run A e no smoke. Falso
positivo de instrumentacao, nao pressao real: zero iteracoes aceitas, nenhum
progress.csv/checkpoint gerado — restore do iter218 permanece limpo. A
tentativa ficou quarentenada em ray_results/h1_runB_iter235_aborted_wd_falsepos
e o log em watchdog_falsepos.log.

Correcao: scripts/watchdog_host_memory.sh agora computa
CONT = memory.current - inactive_file (paridade com docker stats); limites
inalterados (host < 2,5 GB ou container >= 6,5 GiB). Run B relancado 12:27
com preflight aprovado (10,78/10,80 GB, 0 CUDA apps).

### 2026-08-15 13:0x — Run B tentativa 2 abortada (transiente real); emenda do gate

Tentativa 2 (metrica correta): iteracoes 219..221 limpas (this_iter=38.520,
41/42/41 episodios, 0 restarts, reward medio -898 na iter221), mas o watchdog
abortou aos 19,5 min: container saltou 4,0 -> 6,70 GiB entre duas amostras de
1 s na fronteira da iter222. Diagnostico: transiente NORMAL do train step —
o run A valido ja atingira 6,778 GiB de pico global. Gate instantaneo de
6,5 GiB refutado como discriminador.

policy-verifier (emenda): APROVADO COM RESSALVAS. Parametros finais:
- host: abort imediato se MemAvailable < 2,5 GB (inalterado);
- container: abort somente se cont >= 6,5 GiB SUSTENTADO por >= 4,0 s;
- poll nominal 0,5 s; metrica memory.current - inactive_file;
- cap 7g + swap 7g + shm 1536m inalterados; fail-fast interno permanece.
Rejeitadas: elevar limiar para 6,95 GiB (perto demais do cap) e remover o
gate (perderia deteccao de leak sustentado).
Quarentena: iteracoes 219..221 e checkpoint_000000 da tentativa 2
(h1_runB_iter235_aborted_transient) — consistentes, mas descartados por
comparabilidade; restart limpo do iter218 validado.
Teste discriminante da emenda: na 1a fronteira de iteracao acima de 6,5 GiB,
a memoria deve voltar abaixo do limiar em < 4,0 s sem OOM/worker morto.

Tentativa 3 lancada (preflight 10,86/10,86 GB, 0 CUDA apps, disco 37,2 GB),
watchdog emendado em scripts/watchdog_host_memory.sh.

### 2026-08-15 14:11 — Run B tentativa 3 CONCLUIDA; emenda do watchdog validada

TERMINATED (EXIT=0) em 81 min: 17 iteracoes 219..235, todas com delta exato
38.520 (primeiro delta vs iter218 tambem 38.520), episodios > 0, 1 worker,
0 restarts, ts final 9.052.200. Checkpoint iter235 (checkpoint_000003)
validado e persistido em experiment_results/h1_iter235_checkpoint_validation.json:
counters exatos (env 9.052.200, agent 54.313.200 = 6x), blue l2 62,8677
(progrediu de 61,9916), yellow l2 60,9397 inalterada, optimizer 32 tensores.

Teste discriminante da emenda do watchdog: VALIDADO. 27 transientes acima de
6,5 GiB no run inteiro, todos com duracao de 1-2 amostras (< 4 s, um por
fronteira de iteracao), zero aborts, zero OOM. O gate sustentado de 4 s
discrimina corretamente transiente de leak.

Avaliacao final H1 em andamento (container h1_eval, CPU-only): iter235 blue
deterministic vs ckpt3 e ckpt0 (yellow sample), 80 seeds fixos 0..79.
Baselines iter210: 35W/1L/44T vs ckpt3; 46W/0L/34T vs ckpt0.

### 2026-08-15 14:4x — H1 ACEITA: iter235 supera iter210 neste protocolo

Avaliacao final (80 seeds fixos 0..79, blue deterministic, yellow sample,
harness scripts/evaluate_checkpoints_cpu.py, container historico CPU-only):

| confronto | iter210 (baseline) | iter235 (candidato) |
|---|---|---|
| vs ckpt3 | 35W/1L/44T | 49W/0L/31T |
| vs ckpt0 | 46W/0L/34T | 73W/0L/7T |

- Pareado vs ckpt3: +0,188, IC95% [+0,031, +0,344] (t79: LB +0,029) > 0 → ACEITE.
- Pareado vs ckpt0: +0,338, IC95% [+0,212, +0,463] → LB >= 0 → gate nao-regressao PASSA.
- vs ckpt3: 29 melhorias, 14 regressoes, 37 inalterados por seed.

policy-verifier (post-result): APROVADO COM RESSALVAS. H1 ACEITA com conclusao
restrita: "iter235 supera iter210 neste protocolo, para esta unica seed de
treino". iter235 promovido a referencia OFFLINE da campanha (ponto de partida
de H3/extensoes); NAO aprovado para deploy grSim ainda.

Condicoes cumpridas:
- Manifesto imutavel: experiment_results/h1_iter235_manifest_sha256.txt
  (SHA-256 de policy_state.pkl blue/yellow, algorithm_state, params.json,
  harness e dos 4 JSONL; policy_blue iter235 = c88b8174...).
- Baseline filtrado explicito: crossplay_iter210_vs_ckpt3_bdet_ysample_filtered.jsonl
  (80 registros deterministic/yellow-stochastic extraidos do arquivo de 160).
- iter210 e JSONL originais preservados sem sobrescrita.

Ressalvas registradas: pareamento nao e replay identico do oponente (yellow
stochastic depende do estado induzido pelo blue); 1 training seed; retornos
brutos nao comparaveis sem normalizacao por steps.

Pendencias para o gate grSim do iter235 (rodada propria): export de inferencia,
paridade treino vs standalone (strict, mean), gate cartesiano de comandos,
ativacao gradual com modo deterministic (decisao H2), watchdogs de frescor.

### 2026-08-15 — Gate grSim do iter235: APROVADO COM RESSALVAS (operacional)

Execucao autonoma autorizada pelo usuario (testar grSim e seguir decidindo).

Gates executados (artefatos em experiment_results/grsim_gate_iter235/):
- Export de inferencia para volumes/.../campaign2_iter235/checkpoint_000003
  (16 tensores/531.709 params por policy); pesos BIT-EXATOS vs fonte
  (verificacao numpy array_equal em ambas as policies); SHA-256 do export
  persistido em export_sha256.txt.
- Paridade RLlib vs standalone no runtime CPU: INFERENCE_PARITY_OK, 64
  vetores, erro 0.000e+00 em logits, values e deterministic_sample
  (strict=True). Persistido em parity_and_contract.txt.
- Contrato: 16/16 testes (unittest tests.test_grsim_contract) OK.
- Pipeline real (docker-compose.grsim.yml, network host): 3 containers
  healthy; iter235 carregado nos dois times; 3 robos com acoes distintas.
- Janela mean 480 s (monitor_mean_480s.json): 11 episodios completos, 80.024
  pacotes, 1 gol blue (48,02 s); evento yellow t=0,01 s EXCLUIDO como
  artefato de inicializacao (teleporte de kickoff no 1o frame).
- Janela sample 480 s (monitor_sample_480s.json): 1 gol blue (238,84 s).
- Stale: 25 eventos "Visao incompleta/stale; enviando comandos zero" no log
  persistido (logs/policy.log), agrupados em startup/fronteiras de kickoff —
  correcao do relato anterior ("0 stale" contava só stdout do container).
  Watchdog de frescor ATUOU corretamente emitindo zero-command.
- Shutdown: stop_policy.sh limpo; zero-commands no finally (3x por time,
  deploy_policy_grsim.py) + cobertura por teste de contrato.
- Operacional: submodulo rSoccer inicializado no pin 2520207 (v1.2.0) para o
  build do Dockerfile.policy; 3 containers Exited antigos removidos.

policy-verifier: APROVADO COM RESSALVAS. iter235 promovido a checkpoint
OPERACIONAL do deploy grSim em modo mean — rotulado como "operacional e
compativel", SEM alegacao de superioridade no grSim (janelas espelho nao sao
comparacao controlada com ckpt2/sample). Condicoes cumpridas: hash do export
persistido e bit-exatidao verificada; contagem real de stale corrigida;
CHECKPOINT_PATH e ACTION_MODE=mean devem ser SEMPRE explicitos no start
(default do compose segue sample). NAO promover para robo real.

### 2026-08-15 — Pre-registro do run C (H1-ext iter235→260)

Debate idea-debater: APOIO COM MUDANCAS para H1-ext; H3 DESCARTADA ate
evidencia offline de dispersao residual como gargalo (entropia ja caiu de
-0,31 a -1,04 sozinha; mean ja aceito); self-play real adiado para campanha
propria com regra de sync explicita.

policy-verifier (preflight): REPROVADO ate corrigir pre-registro; correcoes
abaixo ADOTADAS antes do lancamento:
- Manifesto preflight imutavel: experiment_results/h1ext_runC_preflight_manifest.txt
  (commit 07d5c1a, hashes de RL_train.py, config, watchdog, harness, rewards
  hist e do checkpoint fonte iter235 blue c88b8174/yellow 0c63a15c).
- Receita identica ao run B; unica variavel = duracao. Restore iter235,
  stop 10.015.200 (iter260), checkpoint final pre-congelado como candidato
  (sem selecao intermediaria).
- Sync da yellow: PERMITIDO (faz parte da receita historica score>0.6). Se
  ocorrer em 236..260: registrar a iteracao, comparar hash da yellow no
  checkpoint final vs 0c63a15c e reclassificar o braco como "continuacao com
  adversario adaptativo" (nao aborta, nao muda o gate esportivo).
- Gate de restore: primeiro resultado deve ter iteration=236 e
  ts=9.090.720 (delta exato 38.520); divergencia aborta o run.
- Regioes de decisao COMPLETAS: primaria vs ckpt3 (baseline iter235
  49W/0L/31T) e nao-regressao vs ckpt0 (73W/0L/7T), 80 seeds fixos 0..79,
  blue deterministic vs yellow sample.
  - ACEITE: pareado vs ckpt3 com IC97,5% LB>0 E ponto vs ckpt0 >=0 E
    derrotas adicionais <=1 (somando os dois confrontos).
  - REJEICAO: ponto vs ckpt3 <=0 OU regressao vs ckpt0 com UB<0 OU
    derrotas adicionais >=4 (limiar material = 5 pp).
  - INCONCLUSIVO: qualquer outra regiao — bloqueia promocao; permite UMA
    extensao confirmatoria ate iter285 tambem com IC97,5% (Bonferroni,
    2 olhares, alpha global 5%). Sem LB>0 em 285, encerra a receita
    conservadora.
- Telemetria adicional obrigatoria: gols pro/contra e derrotas por confronto
  (degradacao defensiva nao pode ser mascarada pelo saldo).
- Gates operacionais inalterados: preflight de recursos (2 medicoes),
  watchdog canonico ativo, fail-fast interno, validacao estrutural e por
  hash do checkpoint final antes da avaliacao.

### 2026-08-15 16:29 — Run C abortado pelo watchdog (pressao de host); retomada C2

Run C: 18 iteracoes 236..253 LIMPAS (delta exato 38.520, episodios>0, 1
worker, 0 restarts; gate de restore iter236 PASSOU). Watchdog matou o
container as 16:29:45 por MemAvailable=2,36 GB < 2,5 GB — pressao dominada
por workload alheio do desktop (VS Code + Firefox ~3,8 GB), com contribuicao
do transiente do treino no segundo final (classificacao do verificador:
pressao MISTA; watchdog agiu conforme desenho; container em 5,35 GiB < 6,5).
Nenhum workload alheio foi interrompido.

Estado validado: checkpoint_000002 = iter250 persistido ANTES do abort
(experiment_results/h1ext_iter250_checkpoint_validation.json: env 9.630.000,
agent 57.780.000 = 6x, blue l2 63,5139 progrediu de 62,8677, yellow 60,9397
INALTERADA, optimizer 32 tensores). Iteracoes 251..253: DESCARTADAS (sem
checkpoint; nao usadas em nenhuma decisao).

policy-verifier (retomada): APROVADO COM RESSALVAS. Emenda aceita como
confirmatoria CONDICIONADA: trajetoria segmentada iter235→250 (run C) +
250→260 (run C2), endpoint iter260 e 80 seeds INALTERADOS; sem avaliacao de
checkpoints intermediarios. Ressalvas adotadas:
- ScoreCounter do sync e recriado vazio no restore (estado dinamico nao
  persistido — igual ao precedente run A→B); se houver sync da yellow em
  251..260 (comparar hash vs 0c63a15c e log "Updating Opponent"), o
  resultado deixa de ser confirmatorio do desenho original sem novo parecer.
- Manifesto C2 persistido ANTES do lancamento:
  experiment_results/h1ext_runC2_preflight_manifest.txt (restore iter250
  blue d05a1b0c/yellow 0c63a15c, hashes de config/reward/watchdog/launcher).
- Relancador automatico scripts/relaunch_runC2_when_free.sh: preflight
  completo em loop (9,5e9 bytes 2x/20s, 0 CUDA apps, VRAM>=10.240, disco
  >=30 GB), restore com glob de cardinalidade 1 + verificacao SHA-256,
  watchdog canonico com kill garantido no exit.
- Gate de restore C2: primeiro resultado deve ter iteration=251,
  ts=9.668.520, agent 58.011.120, delta 38.520.
- RNG de coleta nao fixada: 251..260 do C2 e outra realizacao estocastica
  (variancia, nao vies — descartadas nao influenciaram decisao).

### 2026-08-15 17:58 — Run C2 concluido; gate iter260 INCONCLUSIVO; campanha encerrada

Operacional: 1a tentativa do C2 falhou com EXIT=1 em 18 s por bug do launcher
(--restore com caminho do host em vez de /campaign; trial PPO_Soccer_12b07,
0 iteracoes, sem contaminacao; log em train_failed_hostpath.log). Launcher
corrigido e commitado. Execucao valida (PPO_Soccer_53674): restore limpo do
iter250, 10 iteracoes 251..260 com delta exato 38.520, episodios>0, 1 worker,
0 restarts, 0 aborts, EXIT=0. ZERO "Updating Opponent"; hash yellow no
checkpoint final = 0c63a15c EXATO → condicao confirmatoria satisfeita.
Checkpoint iter260 validado (h1ext_iter260_checkpoint_validation.json):
env 10.015.200, agent 60.091.200 = 6x, blue l2 63,9399, optimizer 32 tensores.

Avaliacao final (80 seeds 0..79, blue det, yellow sample):
| confronto | iter235 (ref) | iter260 |
|---|---|---|
| vs ckpt3 | 49W/0L/31T | 51W/0L/29T |
| vs ckpt0 | 73W/0L/7T | 64W/0L/16T |
- Pareado vs ckpt3: +0,025, IC97,5% [-0,166,+0,216] → LB<=0 (aceite FALHA).
- Pareado vs ckpt0: -0,113, IC97,5% [-0,225,+0,00023] → UB>0 (rejeicao NAO
  dispara); 0 derrotas em 160 episodios.
- Regioes pre-registradas → INCONCLUSIVO. Promocao BLOQUEADA.

policy-verifier (post-result): APROVADO COM RESSALVAS. Condicoes cumpridas:
- iter260 classificado como INCONCLUSIVO (nao REJEITADO);
- iter235 permanece a referencia offline e o checkpoint operacional grSim;
- manifesto pos-resultado persistido
  (experiment_results/h1ext_iter260_postresult_manifest.txt, blue iter260
  8b3d4835, yellow 0c63a15c, JSONLs e script de analise);
- UB nao arredondado registrado: +0,00023;
- tentativa EXIT=1 documentada separadamente da execucao valida.

DECISAO DE ENCERRAMENTO (futilidade/custo, nao rejeicao estatistica):
a extensao permitida ate iter285 NAO sera usada. Justificativa: ganho
marginal pontual caiu de +0,188 (210→235) para +0,025 (235→260); sinal de
regressao vs ckpt0 (teste de sinal pos-hoc nos 17 pares discordantes:
p~0,049 bilateral — analise de sensibilidade, nao gate); custo ~3h com baixa
chance de LB97,5%>0. Registrado que isso NAO prova saturacao nem causa raiz;
qualquer retomada futura ate iter285 consome a UNICA extensao ja prevista.

## Encerramento da campanha 2 — melhor politica: iter235

- MELHOR POLITICA VERIFICADA: iter235 (blue sha256 c88b8174...), checkpoint
  completo em training_runs/h1_runB/.../checkpoint_000003 e export de
  inferencia operacional em
  volumes/dgx_checkpoints/PPO_selfplay_rec/campaign2_iter235/checkpoint_000003.
- Ganhos comprovados sobre o estado inicial da campanha (iter210):
  vs ckpt3 35W→49W (pareado +0,188 IC95% [+0,031,+0,344]);
  vs ckpt0 46W→73W (+0,338 [+0,212,+0,463]); 0 derrotas.
- Ganho de inferencia (H2): modo mean > sample (+0,312 [+0,162,+0,463]);
  deploy grSim validado em mean (paridade erro 0, 16/16 contrato, janelas
  480 s com gol, watchdogs e shutdown zero-command).
- Comando de deploy: CHECKPOINT_PATH=/checkpoints/PPO_selfplay_rec/campaign2_iter235/checkpoint_000003 \
    ACTION_MODE=mean ./start_policy.sh -d   (explicitos SEMPRE).
- NAO promover para robo real (fora do escopo desta campanha).
- Backlog para campanha futura: self-play real com regra de sync explicita
  (yellow congelada desde iter210 e a causa-candidata do platô; exige
  probe de 5 iteracoes + braco controle de mesma duracao + pool de
  avaliacao ampliado com iter235).

---

# Campanha 3 — H-sync: adversario obsoleto como causa do platô (2026-08-15)

## Hipotese e desenho

H-sync: o platô 235→260 foi causado pela yellow congelada no iter210 (50
iteracoes defasada; zero syncs em 211..260). Predicao: treinar a partir do
iter235 com yellow=iter235 restaura o gradiente.

Debate idea-debater: APOIO COM MUDANCAS — todas adotadas:
- yellow CONGELADA em iter235 no braco D (guarda FREEZE_OPPONENT=1 em
  RL_train.py, no-op sem a env var; preserva semantica dos runs A-C2);
- contraste causal primario = D vs C (controle real ja pago, yellow=iter210);
- probe de 5 iteracoes (235→240) com seeds diagnosticos 80..119 ANTES das 25;
  seeds 0..79 reservados para o endpoint iter260;
- restore descartavel validando pesos efetivamente carregados;
- CONTINUAR do probe e gate exploratorio de custo, NAO confirmacao.

## Diario da campanha 3

### 2026-08-15 — Preparacao do braco D e pareceres

- Cirurgia de checkpoint: scripts/make_synced_checkpoint.py copia o iter235 e
  substitui SOMENTE policy_yellow.weights pelos pesos da blue (cloudpickle,
  spec/action dist da yellow preservada). Saida: 16 tensores, 531.709 params,
  yellow==blue bit-exato, l2=62,8677. Artefato:
  training_runs/h3_sync/checkpoint_iter235_ysync (blue c88b8174 INALTERADO,
  yellow 173be554).
- Restore descartavel: scripts/validate_restore_weights.py (PPO 0 workers,
  CPU) → RESTORE_VALIDATION_OK iteration=235, yellow==blue bit-exato nos
  pesos carregados (experiment_results/h3sync_restore_validation.log).
- policy-verifier preflight 1: REPROVADO — 7 condicoes. Atendidas:
  harness ganhou --seed-start (seeds absolutos); dry-run {80,81} verificado
  (h3sync_dryrun_seeds8081.jsonl); criterios exatos pre-registrados
  (D1/D0/Dref, score ±1/0, margens explicitas); C240 validado
  (h3sync_C240_checkpoint_validation.json: iter240, yellow 0c63a15c);
  manifesto v2 completo (h3sync_probe_preflight_manifest_v2.txt).
- policy-verifier preflight 2: INCONCLUSIVO — exigiu persistir restore e
  preflight de recursos. Atendidos: h3sync_restore_validation.log e
  h3sync_probe_resource_preflight.txt (10,22/10,41 GB, 0 CUDA apps, VRAM
  11,4 GiB, disco 40,6 GB — 2 medicoes/20 s PASSAM). JSONLs de avaliacao
  serao um por confronto (chave de retomada nao inclui yellow_checkpoint).
- Run D1 lancado 19:12 (container h3_probeD1, FREEZE_OPPONENT=1, restore
  cirurgico, stop 9.244.800 = iter240, watchdog concorrente com CID ativo
  antes da iter236).

### 2026-08-15 19:37 — Probe D1 concluido (EXIT=0); gates operacionais PASSARAM

- 5 iteracoes 236..240 exatas (delta 38.520, episodios>0, 1 worker, 0
  restarts), ts final 9.244.800; watchdog sem aborts; ZERO "Updating
  Opponent" e ZERO "Sync suprimido" (score interno nunca passou de 0,6).
- Score interno vs espelho subiu monotonicamente: 0,02 → 0,09 → 0,15 →
  0,22 → 0,23 (telemetria; nao e metrica de decisao).
- Checkpoint D240 validado (h3sync_D240_checkpoint_validation.json):
  iter240, env 9.244.800, agent 55.468.800 = 6x, blue l2 63,0270
  (progrediu de 62,8677), yellow l2 62,8677, optimizer 32 tensores.
- Yellow CONGELADA comprovada por conteudo: pesos da yellow do D240 ==
  blue iter235 bit-exato, 16/16 tensores
  (h3sync_D240_yellow_frozen_check.txt). Nota: o hash do pkl difere do
  manifesto (80f46b90 vs 173be554) por re-serializacao do RLlib ao salvar;
  o gate substantivo e a igualdade bit-exata dos pesos, que PASSOU.
- Avaliacoes diagnosticas em andamento (seeds 80..119, JSONL separado por
  confronto): D240/C240 vs ckpt3 e ckpt0, iter235 vs ckpt3.

### 2026-08-15 19:5x — Probe D1: DECISAO CONTINUAR; run D2 lancado (240→260)

Avaliacao diagnostica (seeds 80..119, blue det, yellow sample, JSONL separado
por confronto em experiment_results/h3sync_*_s80.jsonl):
| confronto | W/L/T |
|---|---|
| D240 vs ckpt3 | 26W/0L/14T |
| C240 vs ckpt3 | 20W/0L/20T |
| iter235 vs ckpt3 | 22W/0L/18T |
| D240 vs ckpt0 | 36W/0L/4T |
| C240 vs ckpt0 | 38W/0L/2T |
D1 = +0,150 (>0) | derrotas 0<=0 | D0 = -0,050 (>= -0,05, no limite) |
Dref = +0,100 → DECISAO pre-registrada: CONTINUAR (gate exploratorio de
custo; NAO confirma H-sync). Persistido em h3sync_probe_decision.txt.

policy-verifier (post-result D1 + preflight D2): APROVADO COM RESSALVAS.
Condicoes adotadas antes do lancamento do D2:
- restore descartavel do D240 com refs externas (validador generalizado):
  RESTORE_VALIDATION_OK iteration=240, blue==pkl D240 e yellow==blue iter235
  bit-exato (h3sync_D240_restore_validation.log);
- gates hierarquicos do endpoint codificados em
  scripts/analyze_h3sync_endpoint.py: (1) causal D260-C260 vs ckpt3 com IC t
  pareado 95% (t79=1,9905) LB>0, fail-closed em seed ausente/duplicado/
  metadata; so se passar, (2) promocao cumulativa D260-iter235 LB>0 E ponto
  vs ckpt0 >= -0,05 E derrotas adicionais vs ckpt0 <= 1;
- desvio registrado: gate de hash da yellow substituido por igualdade
  bit-exata de conteudo (re-serializacao do RLlib muda o pkl);
- "Sync suprimido" NAO e falha (comportamento correto do freeze se score>0,6);
- manifesto D2 persistido (h3sync_runD2_preflight_manifest.txt) com commit,
  image ID, launcher integral e hashes; preflight de recursos 10,21/10,19 GB
  PASSA; watchdog com CID no cabecalho do log.

Run D2 lancado 19:57 (container h3_runD2, restore D240, stop 10.015.200 =
iter260, FREEZE_OPPONENT=1). Gate de restore: iter241, ts 9.283.320.

### 2026-08-15 21:40 — Endpoint D260: INCONCLUSIVO; campanha 3 ENCERRADA

Run D2: EXIT=0, 20 iteracoes 241..260 exatas, 0 aborts, ZERO syncs; D260
validado (env 10.015.200, agent 60.091.200, blue l2 63,7350, yellow congelada
bit-exata 16/16; h3sync_D260_checkpoint_validation.json e
h3sync_D260_yellow_frozen_check.txt).

Endpoint (80 seeds holdout 0..79, blue det, yellow sample):
| confronto | W/L/T |
|---|---|
| D260 vs ckpt3 | 54W/0L/26T |
| C260 vs ckpt3 | 51W/0L/29T |
| iter235 vs ckpt3 | 49W/0L/31T |
| D260 vs ckpt0 | 74W/0L/6T |
| iter235 vs ckpt0 | 73W/0L/7T |
GATE 1 causal (D260-C260 vs ckpt3): +0,0375 IC95% [-0,1104,+0,1854] → LB<=0
→ hierarquia PARA → INCONCLUSIVO (ponto >0, nao REJEITADO). Info:
D260-iter235 vs ckpt3 +0,0625 [-0,0892,+0,2142].

policy-verifier (post-result): APROVADO COM RESSALVAS. Condicoes adotadas:
- D260 registrado como "candidato nao promovido; ponto-dominante nos
  comparadores observados; evidencia insuficiente de beneficio causal";
- formulacao oficial: "H-sync NAO confirmada; estimativas pontuais positivas
  neste run, compativeis tambem com efeito nulo ou negativo dentro da
  incerteza observada" (1 seed de treino; C e D confundem hipotese com
  variabilidade de trajetoria; D1=+0,150 selecionou a continuacao, nao e
  confirmacao independente);
- iter235 PERMANECE referencia offline e checkpoint operacional grSim;
- D260 (blue 3a1df217...) e D240 preservados APENAS como candidatos
  exploratorios/membros de pool (sem status de referencia/deploy);
- manifesto pos-resultado com SHA-256 completos persistido
  (h3sync_endpoint_postresult_manifest.txt);
- encerramento por CUSTO/FUTILIDADE, nao rejeicao de H-sync;
- reabertura causal exigiria pares C/D em MULTIPLAS seeds de treino
  pre-fixadas (prolongar so D muda o estimando e nao resolve o confusor).

## Encerramento da campanha 3

- H-sync nao confirmada (sem rejeicao); ganho pontual consistente mas sem
  significancia causal na resolucao deste desenho (meia-largura ~0,15).
- Melhor politica OPERACIONAL: iter235 (inalterada).
- Melhor politica PONTUAL observada: D260 (54W/74W, 0 derrotas) — nao
  promovida; disponivel para pool de avaliacao de campanhas futuras.
- Backlog atualizado: self-play real com regra de sync explicita permanece a
  hipotese estrutural candidata; novo pre-registro deve exigir replicacao
  por seeds de treino e pool ampliado (ckpt0, ckpt3, iter235, D240, D260).
