#!/bin/bash
#
# Script para parar a pipeline grSim + RL Policy
#
# Uso:
#   ./stop_policy.sh
#

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

COMPOSE_FILE="docker-compose.grsim.yml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/scripts/sh/compose_exec.sh"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Verificar se Docker Compose file existe
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Arquivo Docker Compose nao encontrado: $COMPOSE_FILE"
    exit 1
fi

log_info "Parando containers..."
compose_exec -f "$COMPOSE_FILE" down

log_info "Removendo containers parados..."
docker container prune -f > /dev/null 2>&1 || true

log_info "Status dos containers:"
docker ps | grep -E "grsim|rl-policy" || echo "Nenhum container da pipeline rodando"

log_success "Pipeline parada com sucesso!"
