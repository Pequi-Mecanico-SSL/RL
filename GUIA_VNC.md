# Guia de Visualização do grSim via VNC

Este guia explica como visualizar o simulador grSim em tempo real enquanto a policy RL executa.

## Arquivos Disponíveis

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| `docker-compose.grsim.yml` | **Padrão com VNC** (rede bridge) | Quando quer acessar via `localhost:5900` |
| `docker-compose.grsim-vnc.yml` | **VNC com network host** | Quando precisa de UDP multicast (melhor performance) |
| `docker-compose.grsim-headless.yml` | **Sem GUI** | Para treinamento em background |

---

## Opção 1: VNC com Rede Bridge (Recomendado - Mais Fácil)

Esta opção mapeia a porta VNC para `localhost:5900`, facilitando a conexão.

### Iniciar

```bash
cd /home/marcos-paulo/Documentos/RL
docker-compose -f docker-compose.grsim.yml up --build
```

### Conectar via VNC

#### Linux

```bash
# Usando vncviewer (TigerVNC)
vncviewer localhost:5900

# Usando Remmina (GUI)
remmina vnc://localhost:5900

# Usando Vinagre (GNOME)
vinagre localhost:5900
```

#### macOS

```bash
# Terminal
open vnc://localhost:5900

# Ou use o "Screen Sharing" (Compartilhamento de Tela) do macOS:
# Cmd + Espaço → "Screen Sharing" → Digite: vnc://localhost:5900
```

#### Windows

1. Baixe e instale o **TightVNC Viewer** ou **RealVNC Viewer**
2. Abra o viewer
3. Digite: `localhost:5900`
4. Senha (se solicitada): `robocup`

---

## Opção 2: VNC com Network Host (Melhor Performance UDP)

Esta opção usa `network_mode: host` para melhor performance UDP/multicast. O VNC é acessível pelo **IP da máquina**.

### Iniciar

```bash
cd /home/marcos-paulo/Documentos/RL
docker-compose -f docker-compose.grsim-vnc.yml up --build
```

### Descobrir o IP da Máquina

```bash
# Linux
ip addr show | grep "inet " | head -1

# Exemplo de output:
# inet 192.168.1.100/24 brd 192.168.1.255 scope global dynamic eth0
# Use: 192.168.1.100:5900

# macOS
ifconfig | grep "inet " | head -1
```

### Conectar via VNC

Substitua `<IP_DA_MAQUINA>` pelo IP descoberto acima (ex: `192.168.1.100:5900`).

#### Linux

```bash
vncviewer <IP_DA_MAQUINA>:5900

# Exemplo:
vncviewer 192.168.1.100:5900
```

#### macOS

```bash
open vnc://<IP_DA_MAQUINA>:5900

# Exemplo:
open vnc://192.168.1.100:5900
```

#### Windows

1. Abra o VNC Viewer
2. Digite: `<IP_DA_MAQUINA>:5900` (ex: `192.168.1.100:5900`)
3. Senha: `robocup`

---

## Configurações de VNC

As configurações são definidas via variáveis de ambiente no docker-compose:

```yaml
environment:
  - VNC_PASSWORD=robocup      # Senha para acesso (opcional)
  - VNC_GEOMETRY=1920x1080    # Resolução da tela
  - VNC_DEPTH=24              # Profundidade de cor
```

### Para remover a senha:

Comente ou remova a linha `VNC_PASSWORD` no arquivo docker-compose.

### Para mudar a resolução:

Altere `VNC_GEOMETRY` para o valor desejado:
- `1280x720` - HD
- `1920x1080` - Full HD (padrão)
- `2560x1440` - 2K

---

## Clientes VNC Recomendados

### Linux

```bash
# Ubuntu/Debian
sudo apt install tigervnc-viewer remmina vinagre

# Fedora
sudo dnf install tigervnc remmina vinagre

# Arch
sudo pacman -S tigervnc remmina
```

### macOS

- **Screen Sharing** (incluído no macOS)
- **RealVNC Viewer** (download gratuito)
- **TightVNC** (via Homebrew: `brew install tiger-vnc`)

### Windows

- **TightVNC** (https://www.tightvnc.com/)
- **RealVNC Viewer** (https://www.realvnc.com/)
- **UltraVNC** (https://uvnc.com/)

---

## Resolução de Problemas

### "Connection refused" ao conectar VNC

```bash
# Verificar se container está rodando
docker ps | grep grsim

# Verificar logs
docker-compose -f docker-compose.grsim.yml logs grsim

# Verificar se porta 5900 está em uso
sudo lsof -i :5900
sudo netstat -tulnp | grep 5900
```

### Tela preta no VNC

```bash
# Reiniciar o container
docker-compose -f docker-compose.grsim.yml restart grsim

# Verificar se grSim iniciou corretamente
docker-compose -f docker-compose.grsim.yml logs -f grsim
```

### UDP não funciona (robôs não respondem)

Se usar a opção com rede bridge (Opção 1) e os robôs não se moverem:

```bash
# Tente a Opção 2 (network host)
docker-compose -f docker-compose.grsim-vnc.yml up --build
```

### Performance lenta no VNC

1. Reduza a resolução no docker-compose:
   ```yaml
   - VNC_GEOMETRY=1280x720
   ```
2. Use conexão cabeada em vez de Wi-Fi
3. Feche outros programas pesados

---

## Comandos Úteis

### Parar a Pipeline

```bash
# Opção bridge
docker-compose -f docker-compose.grsim.yml down

# Opção network host
docker-compose -f docker-compose.grsim-vnc.yml down
```

### Ver Logs em Tempo Real

```bash
# Logs do grSim
docker-compose -f docker-compose.grsim.yml logs -f grsim

# Logs da policy
docker-compose -f docker-compose.grsim.yml logs -f rl-policy
```

### Reiniciar Apenas o Simulador

```bash
docker-compose -f docker-compose.grsim.yml restart grsim
```

---

## Exemplo Completo: Primeira Execução

```bash
# 1. Ir para o diretório
cd /home/marcos-paulo/Documentos/RL

# 2. Criar diretório de logs
mkdir -p logs

# 3. Iniciar com VNC (rede bridge - mais fácil)
docker-compose -f docker-compose.grsim.yml up --build

# 4. Em outro terminal, conectar VNC
vncviewer localhost:5900
# Senha: robocup

# 5. Para parar (Ctrl+C no terminal do docker-compose)
# Ou:
docker-compose -f docker-compose.grsim.yml down
```

---

## Alternativas ao VNC

### Gravar Vídeo

Se o VNC não funcionar bem, você pode gravar a tela:

```bash
# No anfitrião, gravar tela
ffmpeg -f x11grab -r 30 -s 1920x1080 -i :0 -vcodec libx264 output.mp4
```

### Web Interface (se disponível)

Algumas versões do grSim podem ter interface web. Verifique:
- http://localhost:8080 (se configurado)

### X11 Forwarding (Linux/Mac avançado)

Requer configuração adicional do X11 no Docker.

---

## Resumo Visual

```
┌─────────────────────────────────────────────┐
│            SUA MÁQUINA (HOST)              │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   VNC Viewer                        │   │
│  │   localhost:5900  ou  IP:5900       │   │
│  └──────────────┬──────────────────────┘   │
│                 │                           │
│                 │ TCP 5900                  │
│                 ▼                           │
│  ┌─────────────────────────────────────┐   │
│  │   Docker Container: grSim           │   │
│  │   ┌─────────────────────────────┐   │   │
│  │   │   grSim + VNC Server        │   │   │
│  │   │   Porta 5900                │   │   │
│  │   │   Porta 20011 (UDP)         │   │   │
│  │   └─────────────────────────────┘   │   │
│  └──────────────┬──────────────────────┘   │
│                 │ UDP 20011                 │
│                 ▼                           │
│  ┌─────────────────────────────────────┐   │
│  │   Docker Container: rl-policy       │   │
│  │   (sua policy RL treinada)          │   │
│  └─────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Próximos Passos

1. ✅ Conecte-se ao VNC
2. ✅ Observe o grSim rodando
3. ✅ Veja os robôs se movendo conforme a policy
4. ✅ Ajuste a resolução se necessário
5. ✅ Divirta-se vendo sua IA jogar futebol!

Para dúvidas, consulte o [GRSIM_DEPLOY_GUIDE.md](GRSIM_DEPLOY_GUIDE.md) ou [GUIA_EXECUCAO_GRSIM.md](GUIA_EXECUCAO_GRSIM.md).
