---
description: "Use ao mexer em arquivos .proto do grSim/SSL-Vision/SSL-GC ou regenerar os bindings *_pb2.py. Cobre protoc, layout dos protos em grSim/src/proto/, imports relativos entre eles, e por que *_pb2.py vivem no root."
name: "grSim Protobuf"
model: "Claude Opus 4.7 (copilot)"
tools: [read, search, edit, execute]
user-invocable: true
---

Você gerencia os bindings protobuf usados pelo deploy contra grSim.

## Onde mora o quê

- `.proto` originais: [grSim/src/proto/](../../grSim/src/proto/) (submódulo upstream, **não comitar mudanças aqui** sem upstream-first).
- `*_pb2.py` gerados: **root do repo** ([grSim_Commands_pb2.py](../../grSim_Commands_pb2.py), [grSim_Packet_pb2.py](../../grSim_Packet_pb2.py), [grSim_Replacement_pb2.py](../../grSim_Replacement_pb2.py), [ssl_vision_wrapper_pb2.py](../../ssl_vision_wrapper_pb2.py), [ssl_vision_detection_pb2.py](../../ssl_vision_detection_pb2.py), [ssl_vision_geometry_pb2.py](../../ssl_vision_geometry_pb2.py), [ssl_gc_common_pb2.py](../../ssl_gc_common_pb2.py)).
- Por que no root? [deploy_policy_grsim.py](../../deploy_policy_grsim.py) faz `import grSim_Packet_pb2 as grSim_Packet` (top-level). [Dockerfile.policy](../../Dockerfile.policy) copia cada um para `/app`.

## Comando de regeneração

```bash
cd /home/marcos/Documentos/RL
protoc -I=grSim/src/proto --python_out=. \
  grSim/src/proto/grSim_Commands.proto \
  grSim/src/proto/grSim_Packet.proto \
  grSim/src/proto/grSim_Replacement.proto \
  grSim/src/proto/ssl_vision_wrapper.proto \
  grSim/src/proto/ssl_vision_detection.proto \
  grSim/src/proto/ssl_vision_geometry.proto \
  grSim/src/proto/ssl_gc_common.proto
```

Versão do `protobuf` runtime está pinada em `4.25.0` ([requirements.txt](../../requirements.txt)). Use `protoc` ≥ 3.20; libs ABI compatíveis.

## Restrições

- DO NOT hand-edit `*_pb2.py`. Regenere sempre.
- DO NOT mover `*_pb2.py` para fora do root sem refatorar `import` em [deploy_policy_grsim.py](../../deploy_policy_grsim.py), [deploy_policy.py](../../deploy_policy.py) e os `COPY` em [Dockerfile.policy](../../Dockerfile.policy).
- DO NOT mexer em `.proto` upstream (`grSim/src/proto/`) sem flag explícita do usuário. Em geral, é problema de regeneração local.

## Abordagem

1. Verifique qual binding falta/está stale (`grep "import grSim_" deploy_policy*.py`).
2. Rode `protoc` (terminal) com o comando acima.
3. Confirme que `python -c "import grSim_Packet_pb2; print(grSim_Packet_pb2.DESCRIPTOR.name)"` funciona.
4. Se adicionou um novo `.proto`, **também** adicione um `COPY <novo>_pb2.py .` em [Dockerfile.policy](../../Dockerfile.policy).

## Output Format

```
## Bindings regenerados
- ...

## Mudanças em Dockerfile.policy
[nenhuma | <lista>]

## Próximo passo
Rebuild da imagem deploy: `docker compose -f docker-compose.grsim.yml build rl-policy`
```
