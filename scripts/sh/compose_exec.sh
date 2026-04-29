# Preferir Docker Compose V2 (`docker compose`).
# O compose v1 (Python 1.29.x) quebra com KeyError: 'ContainerConfig' ao recriar
# containers com imagens geradas por Docker/BuildKit recentes.
compose_exec() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}
