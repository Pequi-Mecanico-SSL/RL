#!/usr/bin/env bash
#
# Sobe grSim + rl-policy com Docker Compose (prefere `docker compose` v2).
#
# Uso:
#   ./start_policy.sh
#   ./start_policy.sh -d
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
COMPOSE_FILE="docker-compose.grsim.yml"

# shellcheck source=/dev/null
source "$ROOT/scripts/sh/compose_exec.sh"

# `exec` nao funciona com funcoes shell; chamamos diretamente.
compose_exec -f "$COMPOSE_FILE" up --build "$@"
