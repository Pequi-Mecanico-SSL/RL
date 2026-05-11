---
description: "Regras para editar arquivos .proto do grSim/SSL-Vision/SSL-GC e os bindings *_pb2.py. Use sempre que tocar algum .proto ou *_pb2.py."
applyTo: "**/*.proto, **/*_pb2.py, Dockerfile.policy"
---

# Protobuf grSim — Regras

- **Nunca hand-edit `*_pb2.py`.** Regenere com `protoc` a partir de [grSim/src/proto/](../../grSim/src/proto/).
- **Os `*_pb2.py` vivem no root do repo** porque [deploy_policy_grsim.py](../../deploy_policy_grsim.py) os importa como top-level (`import grSim_Packet_pb2`). Mover exige refatorar imports **e** [Dockerfile.policy](../../Dockerfile.policy).
- **`grSim/src/proto/*.proto` é submódulo upstream.** Não commite mudanças locais — abra PR em https://github.com/Pequi-Mecanico-SSL/rSoccer ou no upstream do grSim primeiro.
- **`protobuf` runtime pinned em `4.25.0`** ([requirements.txt](../../requirements.txt)). Não bumpar sem regenerar todos os `*_pb2.py` contra a mesma ABI.
- Ao adicionar um novo `.proto`: também adicione `COPY <novo>_pb2.py .` em [Dockerfile.policy](../../Dockerfile.policy), senão a imagem deploy quebra com `ImportError`.

Comando de regeneração canônico:
```bash
protoc -I=grSim/src/proto --python_out=. grSim/src/proto/*.proto
```

Quando em dúvida, use o subagente `grsim-proto`.
