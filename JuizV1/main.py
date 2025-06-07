import pygame
from field import SSLRenderField
from ball import Ball
from robot import SSLRobot
from utils import COLORS
from game_logic import get_ball_possession, update_last_touch, is_goal, check_ball_out_of_play, check_ball_fundo

# Inicialização do pygame
pygame.init()

# Criação do campo e da janela
field = SSLRenderField()
screen = pygame.display.set_mode(field.window_size)
pygame.display.set_caption("Simulação Futebol de Robôs")
clock = pygame.time.Clock()

# Escala usada
scale = field.scale

# Definir limites do campo para os robôs
min_x = scale * SSLRenderField.margin
max_x = field.screen_width - scale * SSLRenderField.margin
min_y = scale * SSLRenderField.margin
max_y = field.screen_height - scale * SSLRenderField.margin
robot_bounds = (min_x - 25, max_x + 25, min_y - 25, max_y + 25)

# Limites bola
ball_bounds = (min_x + 10, max_x - 10, min_y, max_y)

# Inicializa bola
ball = Ball(x=field.center_x, y=field.center_y, scale=scale)

# Inicializa robôs
robots = []
# Time azul
robots += [
    SSLRobot(x=field.center_x - 1.5 * scale, y=field.center_y - scale, direction=0, scale=scale, id=0, team_color=COLORS["BLUE"]),
    SSLRobot(x=field.center_x - 1.5 * scale, y=field.center_y,         direction=0, scale=scale, id=1, team_color=COLORS["BLUE"]),
    SSLRobot(x=field.center_x - 1.5 * scale, y=field.center_y + scale, direction=0, scale=scale, id=2, team_color=COLORS["BLUE"]),
]
# Time vermelho
robots += [
    SSLRobot(x=field.center_x + 1.5 * scale, y=field.center_y - scale, direction=180, scale=scale, id=0, team_color=COLORS["RED"]),
    SSLRobot(x=field.center_x + 1.5 * scale, y=field.center_y,         direction=180, scale=scale, id=1, team_color=COLORS["RED"]),
    SSLRobot(x=field.center_x + 1.5 * scale, y=field.center_y + scale, direction=180, scale=scale, id=2, team_color=COLORS["RED"]),
]
controlled_robot = robots[4]  # Robô azul que será controlado com o teclado
# Velocidades de movimento
speed = 1.5
rotation_speed = 2

# Quem tocou por último
last_touch_info = (None, None)
# Adicionado para evitar prints redundantes da posse de bola
previous_possession_info = (None, None)
# Adicionado para evitar prints redundantes do último toque
previous_last_touch_info = (None, None)

goal_posts_info = {
    "LEFT": {
        "x_min": field.margin - field.goal_depth,
        "x_max": field.margin,
        "y_min": (field.screen_height - field.goal_width) / 2,
        "y_max": (field.screen_height - field.goal_width) / 2 + field.goal_width
    },
    "RIGHT": {
        "x_min": field.screen_width - field.margin,
        "x_max": field.screen_width - field.margin + field.goal_depth,
        "y_min": (field.screen_height - field.goal_width) / 2,
        "y_max": (field.screen_height - field.goal_width) / 2 + field.goal_width
    }
}

# Loop principal
running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Leitura de teclas pressionadas
    keys = pygame.key.get_pressed()
    vx = vy = vtheta = 0

    if keys[pygame.K_w]:
        vy -= speed
    if keys[pygame.K_s]:
        vy += speed
    if keys[pygame.K_a]:
        vx -= speed
    if keys[pygame.K_d]:
        vx += speed
    if keys[pygame.K_q]:
        vtheta -= rotation_speed
    if keys[pygame.K_e]:
        vtheta += rotation_speed

    # Aplica movimento ao robô controlado
    controlled_robot.move(vx, vy, vtheta, bounds=robot_bounds)

    # Lógica de posse de bola (antes do loop de robôs, se você quiser a posse atual)
    current_possession_id, current_possession_team_color = get_ball_possession(ball, robots)
    if (current_possession_id, current_possession_team_color) != previous_possession_info:
        if current_possession_id is not None:
            if current_possession_team_color == COLORS["BLUE"]:
                print(f"Posse da bola: Time Azul, Robô ID: {current_possession_id}")
            elif current_possession_team_color == COLORS["RED"]:
                print(f"Posse da bola: Time Vermelho, Robô ID: {current_possession_id}")
        else:
            print("Posse da bola: Livre")
        previous_possession_info = (current_possession_id, current_possession_team_color)


    # Limpa tela e desenha campo
    field.draw(screen)

    colliding_robot_this_frame = None # Variável para armazenar o robô que colidiu
    # Atualiza e desenha robôs
    for robo in robots:
        robo.draw(screen)
        # Exemplo de movimento para testar (apenas para robôs não controlados)
        if robo != controlled_robot:
            robo.move(vx=0, vy=0, vtheta=0.5)

        # Colisão com a bola
        if ball.check_collision(robo):
            # Se houve colisão, bounce_off retornará o robô
            if ball.bounce_off(robo): # Chame bounce_off e verifique se retornou um robô
                colliding_robot_this_frame = robo # Armazene o robô que colidiu

    # Atualiza a física da bola UMA VEZ por frame, APÓS todas as colisões
    ball.update()

    # Atualiza o último toque, passando o robô que colidiu se houver
    new_last_touch_info = update_last_touch(ball, robots, last_touch_info, goal_posts_info, collided_robot=colliding_robot_this_frame)
    if new_last_touch_info != last_touch_info:
        last_touch_info = new_last_touch_info
        if last_touch_info[0] is not None:
            if last_touch_info[1] == COLORS["BLUE"]:
                print(f"Último toque: Time Azul, Robô ID: {last_touch_info[0]}")
            elif last_touch_info[1] == COLORS["RED"]:
                print(f"Último toque: Time Vermelho, Robô ID: {last_touch_info[0]}")
        else:
            print("Último toque: Nenhum (ou Trave)")

    goal_side = is_goal(ball, field)
    if goal_side:
        print(f"Gol do lado {goal_side}!")
        ball.x = field.center_x
        ball.y = field.center_y
        ball.vx = 0
        ball.vy = 0
        previous_possession_info = (None, None)
        last_touch_info = (None, None)
        previous_last_touch_info = (None, None) # Certifique-se de resetar este também

    else:
        # Se não foi gol, verifica se a bola saiu de campo (lateral ou linha de fundo não gol)
        ball_out_of_play = check_ball_out_of_play(ball, field, ball_bounds) #
        if ball_out_of_play:
            previous_possession_info = (None, None) #
            last_touch_info = (None, None) #
            previous_last_touch_info = (None, None) #

        ball_fundo = check_ball_fundo(ball, field, ball_bounds, last_touch_info[1])
        if ball_fundo:
            previous_possession_info = (None, None) #
            last_touch_info = (None, None) #
            previous_last_touch_info = (None, None)

    # Desenha bola
    ball.draw(screen)

    pygame.display.flip()

pygame.quit()
