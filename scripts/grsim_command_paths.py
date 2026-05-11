#!/usr/bin/env python3
"""
Referencia: portas UDP do grSim vs formato protobuf (avaliacao RobotControl vs grSim_Packet).

Implementacao atual do deploy (deploy_policy_grsim.py na raiz do repo):
  - Comandos: UDP 20011, mensagem grSim_Packet / grSim_Commands (isteamyellow).
  - Estado: multicast 224.5.23.2, porta VISION_PORT (padrao grSim 10020), SSL_WrapperPacket.

Caminhos alternativos no codigo-fonte do grSim (grSim/src/sslworld.cpp):
  - 10301 / 10302: RobotControl (protobuf SSL simulation), um socket por time.
  - 10300: SimulatorCommand / SimulatorResponse (teleporte, parte da config).

Migrar para 10301/10302 exigiria serializar RobotControl + RobotCommand +
MoveGlobalVelocity (ou wheel/local), nao reutilizar grSim_Robot_Command.
Veja grSim/src/proto/ssl_simulation_robot_control.proto.

Executar: python3 scripts/grsim_command_paths.py
"""

from __future__ import annotations

ROWS = [
    ("Vision (multicast)", "224.5.23.x", "10020 (padrao configwidget)", "SSL_WrapperPacket / detection"),
    ("Command (legacy)", "0.0.0.0", "20011", "grSim_Packet -> grSim_Commands"),
    ("Blue SSL control", "0.0.0.0", "10301", "RobotControl"),
    ("Yellow SSL control", "0.0.0.0", "10302", "RobotControl"),
    ("Simulation control", "0.0.0.0", "10300", "SimulatorCommand"),
]


def main() -> None:
    print("grSim UDP endpoints vs protobuf (referencia)\n")
    print(f"{'Papel':<22} {'Bind':<12} {'Porta':<8} {'Protobuf'}")
    for row in ROWS:
        print(f"{row[0]:<22} {row[1]:<12} {row[2]:<8} {row[3]}")


if __name__ == "__main__":
    main()
