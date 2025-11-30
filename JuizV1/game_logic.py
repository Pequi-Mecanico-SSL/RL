import math
import numpy as np
from utils import COLORS
from field import SSLRenderField

def is_goal(ball, field):
    # Medidas já escaladas
    goal_top = (field.screen_height - field.goal_width) / 2
    goal_bottom = goal_top + field.goal_width

    # Gol à esquerda
    if field.margin - field.goal_depth <= ball.x <= field.margin:
        if goal_top <= ball.y <= goal_bottom:
            return "LEFT"

    # Gol à direita
    right_goal_x = field.screen_width - field.margin
    if right_goal_x <= ball.x <= right_goal_x + field.goal_depth:
        if goal_top <= ball.y <= goal_bottom:
            return "RIGHT"

    return None


def get_ball_possession(ball, robots, possession_radius_scale=3):
    """
    Determina qual robô tem a posse da bola ou se a bola está livre.

    Args:
        ball (Ball): O objeto da bola.
        robots (list): Uma lista de objetos Robot.
        possession_radius_scale (float): Fator de escala para o raio de posse da bola
                                         em relação ao tamanho do robô.

    Returns:
        tuple: (robot_id, team_color) do robô com a posse, ou (None, None) se a bola estiver livre.
    """
    closest_robot = None
    min_distance = float('inf')

    for robot in robots:
        distance = math.hypot(ball.x - robot.x, ball.y - robot.y)
        if distance < min_distance:
            min_distance = distance
            closest_robot = robot

    if closest_robot:
        # Define a zona de domínio do robô como um pouco maior que seu próprio tamanho
        # Isso pode ser ajustado para simular o "controle" da bola
        possession_threshold = closest_robot.size * possession_radius_scale
        if min_distance <= possession_threshold:
            return closest_robot.id, closest_robot.team_color
    
    return None, None # Bola livre

def check_ball_direction_change(ball):
    """
    Verifica se a direção da bola mudou significativamente.

    Args:
        ball (Ball): O objeto da bola.

    Returns:
        bool: True se a direção mudou, False caso contrário.
    """
    # Vetor de velocidade atual
    v_current = np.array([ball.vx, ball.vy])
    # Vetor de velocidade anterior
    v_previous = np.array([ball.pvx, ball.pvy])

    # Se a bola estava parada e agora está se movendo, ou vice-versa, é uma mudança.
    # Evita dividir por zero em magnitudes nulas.
    if (np.linalg.norm(v_current) < 0.1 and np.linalg.norm(v_previous) < 0.1): # Ambos parados ou quase
        return False
    if (np.linalg.norm(v_current) > 0.1 and np.linalg.norm(v_previous) < 0.1) or \
       (np.linalg.norm(v_current) < 0.1 and np.linalg.norm(v_previous) > 0.1):
        return True # Começou a mover ou parou

    # Calcula o produto escalar e as magnitudes para o ângulo
    dot_product = np.dot(v_current, v_previous)
    magnitude_current = np.linalg.norm(v_current)
    magnitude_previous = np.linalg.norm(v_previous)

    # Evita divisão por zero se alguma magnitude for zero (embora já tratado acima, para segurança)
    if magnitude_current == 0 or magnitude_previous == 0:
        return False # Se um dos vetores é zero, a direção não é bem definida para comparação

    # Calcula o cosseno do ângulo entre os vetores
    cos_theta = dot_product / (magnitude_current * magnitude_previous)

    # Garante que cos_theta esteja no intervalo [-1, 1] devido a possíveis erros de ponto flutuante
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calcula o ângulo em graus
    angle = np.degrees(np.arccos(cos_theta))
    
    # Define um limiar para considerar uma "mudança de direção"
    # Você pode ajustar este valor (por exemplo, 5 graus, 10 graus)
    direction_change_threshold = 1 # graus

    return abs(angle) > direction_change_threshold


def update_last_touch(ball, robots, current_last_touch_info, goal_posts_info, possession_radius_scale=1.5, collided_robot=None):
    """
    Atualiza a informação do último toque na bola.

    Args:
        ball (Ball): O objeto da bola.
        robots (list): Uma lista de objetos Robot.
        current_last_touch_info (tuple): (robot_id, team_color) do último tocador.
        goal_posts_info (dict): Informações sobre as traves (ex: {side: (x_min, x_max, y_min, y_max)})
                                para detectar toque na trave.
        possession_radius_scale (float): Fator de escala para o raio de posse da bola.
        collided_robot (Robot): O robô que acabou de colidir com a bola neste frame, se houver.

    Returns:
        tuple: (robot_id, team_color) do novo último tocador.
    """
    # 1. Detecção de toque direto por colisão (mais prioritário)
    if collided_robot:
        return collided_robot.id, collided_robot.team_color

    # 2. Detecção por posse de bola após uma mudança de direção (fallback)
    #    Isso lida com situações onde a colisão não foi capturada diretamente,
    #    mas a bola mudou de direção e um robô está em posse.
    if check_ball_direction_change(ball):
        robot_id, team_color = get_ball_possession(ball, robots, possession_radius_scale)
        if robot_id is not None:
            return robot_id, team_color
        else:
            # Se a direção mudou, mas nenhum robô tem a posse, pode ter sido a trave
            # Implemente a lógica de detecção de colisão com a trave aqui
            # Por enquanto, retorna None, None para indicar que o último toque não é um robô.
            return None, None # Se a bola mudou de direção e não tem posse, o último toque é "ninguém" ou trave
    
    # 3. Se nada mudou, o último tocador permanece o mesmo
    return current_last_touch_info


# Definindo variáveis e limites para função de lateral e linha de fundo
field = SSLRenderField()
scale = field.scale

min_x = scale * SSLRenderField.margin
max_x = field.screen_width - scale * SSLRenderField.margin
min_y = scale * SSLRenderField.margin
max_y = field.screen_height - scale * SSLRenderField.margin

def check_ball_out_of_play(ball, field, ball_bounds):
    min_x + 10, max_x - 10, min_y, max_y == ball_bounds #
    
    # Verifica se a bola saiu pelas laterais
    if ball.y < min_y or ball.y > max_y: #
        print("Lateral!") #
        ball.y = min_y if ball.y < min_y else max_y #
        ball.vx = 0 #
        ball.vy = 0 #
        return True # Indica que a bola saiu de jogo
    return False # A bola não saiu de jogo

def check_ball_fundo(ball, field, ball_bounds, time_ult_toque):
    min_x + 10, max_x - 10, min_y, max_y == ball_bounds
    center_y = field.center_y
    TIME_AZUL = COLORS["BLUE"]
    TIME_RED = COLORS["RED"]
    # Verifica se a bola saiu pelas linhas de fundo (não sendo gol)

 # =================== LÓGICA BOLA SAINDO PELA LINHA DE FUNDO =================== 
 # A lógica esta baseada em que o time AZUL defende o lado ESQUERDO e o RED o lado DIREITO

    if ball.x < min_x or ball.x > max_x:
        # SE A BOLA SAIU ELA TEM QUE PARAR
        ball.vx = 0 #
        ball.vy = 0  #
        
        # === Esquerda INFERIOR ===
        # TIME AZUL FOI O ULTIMO TOQUE == ESCANTEIO RED
        if ball.x < min_x and ball.y > center_y and time_ult_toque == TIME_AZUL:
            print("ESCANTEIO ESQUERDA INFERIOR PARA RED")
            ball.y = field.screen_height - field.margin # <<< CORREÇÃO APLICADA
            ball.x = field.margin
        # TIME RED FOI O ULTIMO TOQUE == TIRO DE META AZUL
        if ball.x < min_x and ball.y > center_y and time_ult_toque == TIME_RED:
            print("TIRO DE META PARA AZUL")
            ball.x = field.margin + field.penalty_length / 2
            ball.y = field.screen_height / 2
        
        # === Esquerda SUPERIOR ===
        # TIME AZUL FOI O ULTIMO TOQUE == ESCANTEIO RED
        if ball.x < min_x and ball.y < center_y and time_ult_toque == TIME_AZUL:
            print("ESCANTEIO ESQUERDA SUPERIOR PARA RED")
            ball.y = field.margin
            ball.x = field.margin
        # TIME RED FOI O ULTIMO TOQUE == TIRO DE META AZUL
        if ball.x < min_x and ball.y < center_y and time_ult_toque == TIME_RED:
            print("TIRO DE META PARA AZUL")
            ball.x = field.margin + field.penalty_length / 2
            ball.y = field.screen_height / 2

        # === Direita INFERIOR ===
        # TIME RED ULTIMO TOQUE = ESCANTEIO AZUL
        if ball.x > max_x and ball.y > center_y and time_ult_toque == TIME_RED:
            print("ESCANTEIO DIREITA INFERIOR PARA AZUL")
            ball.y = field.screen_height - field.margin # <<< CORREÇÃO APLICADA
            ball.x = field.screen_width - field.margin
        # TIME AZUL ULTIMO TOQUE = TIRO DE META RED
        if ball.x > max_x and ball.y > center_y and time_ult_toque == TIME_AZUL:
            print("TIRO DE META PARA RED")
            ball.x = field.screen_width - field.margin - field.penalty_length / 2 
            ball.y = field.screen_height / 2 

        # === Direita SUPERIOR ===
        # TIME RED ULTIMO TOQUE = ESCANTEIO AZUL
        if ball.x > max_x and ball.y < center_y and time_ult_toque == TIME_RED:
            print("ESCANTEIO DIREITA SUPERIOR PARA AZUL")
            ball.y = field.margin #
            ball.x = field.screen_width - field.margin
        # TIME AZUL ULTIMO TOQUE = TIRO DE META RED
        if ball.x > max_x and ball.y < center_y and time_ult_toque == TIME_AZUL:
            print("TIRO DE META PARA RED")
            ball.x = field.screen_width - field.margin - field.penalty_length / 2 
            ball.y = field.screen_height / 2
            
            
        return True # Indica que a bola saiu de jogo
    return False # A bola não saiu de jogo