---
description: "Regenera todos os bindings *_pb2.py a partir de grSim/src/proto/*.proto usando protoc."
name: "Regenerate grSim Protos"
agent: "grsim-proto"
model: "Claude Opus 4.7 (copilot)"
---

Regenere os `*_pb2.py` no root do repo a partir dos `.proto` do submódulo grSim.

1. Verifique que `protoc` está disponível: `protoc --version` (≥ 3.20).
2. Liste os `.proto` realmente usados em [grSim/src/proto/](../../grSim/src/proto/) que viram `*_pb2.py` no root: `grSim_Commands`, `grSim_Packet`, `grSim_Replacement`, `ssl_vision_wrapper`, `ssl_vision_detection`, `ssl_vision_geometry`, `ssl_gc_common`.
3. Rode (a partir do root do repo):
   ```bash
   protoc -I=grSim/src/proto --python_out=. \
     grSim/src/proto/grSim_Commands.proto \
     grSim/src/proto/grSim_Packet.proto \
     grSim/src/proto/grSim_Replacement.proto \
     grSim/src/proto/ssl_vision_wrapper.proto \
     grSim/src/proto/ssl_vision_detection.proto \
     grSim/src/proto/ssl_vision_geometry.proto \
     grSim/src/proto/ssl_gc_common.proto
   ```
4. Sanity check: `python -c "import grSim_Packet_pb2, ssl_vision_wrapper_pb2; print('OK')"`.
5. Lembre o usuário de **rebuild** da imagem deploy: `docker compose -f docker-compose.grsim.yml build rl-policy rl-policy-yellow`.

Se algum `.proto` novo aparecer, sinalize que também é necessário adicionar `COPY <novo>_pb2.py .` em [Dockerfile.policy](../../Dockerfile.policy).
