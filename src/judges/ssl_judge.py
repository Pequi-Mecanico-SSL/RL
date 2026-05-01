import numpy as np
import math
from src.objects import Frame, Ball, Robot, Field, InitialPosition
from scipy.spatial import KDTree

class Judge():
    def __init__(self, field: Field, init_pos: dict, n_robots_blue=3, n_robots_yellow=3,
                 possession_radius_scale: float=3, direction_change_threshold: float=1,
                 left="blue", kickoff=True
        ):
        self.field = field
        self.last_frame = None
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        self.ball_possession: str = None

        self.last_touch: str = None
        self.possession_radius_scale = possession_radius_scale
        self.direction_change_threshold = direction_change_threshold #in degrees
        self.left = left

        self.init_pos = InitialPosition(**init_pos)
        self.frame = self._get_initial_positions_frame("kickoff")
        self.historical_ball_positions = set()
        self.is_kickoff = kickoff

    def judge(self, frame) -> tuple[str, dict]:
        """
        Método que executa o julgamento do juiz, verificando a posse de bola, último toque e se houve gol, lateral ou linha de fundo.
        :return: tuple - (status, infos)
            status: str - Indica o status do jogo, podendo ser "left_goal", "right_bottom_line", "sideline" ou None
            infos: dict - Informações adicionais sobre a posse de bola e último toque
        """

        self.last_frame = self.frame
        self.frame = frame

        self._update_ball_possession()
        self._update_last_touch()
        self._update_offenses()
        infos = {
            "ball_possession": self.ball_possession,
            "last_touch": self.last_touch,
            "offenses": self.offenses
        }

        goal = self._check_goal()
        bottom_line = self._check_bottom_line()
        sideline_line = self._check_sideline()

        # status = None indicando que não houve nenhum evento relevante
        status = goal or bottom_line or sideline_line or None

        return status, infos   
    

    def _check_goal(self) -> str|None:
        # Medidas já escaladas
        ball = self.frame.ball
        half_len = self.field.length/2 
        goal_top  = self.field.goal_width / 2
        goal_bottom = -self.field.goal_width / 2

        if ball.x > half_len and goal_bottom <= ball.y <= goal_top:
            return "RIGHT_GOAL"

        if ball.x < -half_len and goal_bottom <= ball.y <= goal_top:
            return "LEFT_GOAL"
        
        return None
    
    def _check_bottom_line(self) -> str|None:
        """
        Verifica se a bola saiu pela linha de fundo
        :return: str - "right_bottom_line" ou "left_bottom_line" se a bola saiu pela linha de fundo, None caso contrário
        """
        ball = self.frame.ball
        half_len = self.field.length / 2
        half_wid = self.field.width / 2

        if ball.x > half_len and abs(ball.y) < half_wid:
            return "RIGHT_BOTTOM_LINE"
        
        if ball.x < -half_len and abs(ball.y) < half_wid:
            return "LEFT_BOTTOM_LINE"
        
        return None
    
    def _check_sideline(self) -> str|None:
        """
        Verifica se a bola saiu pela lateral
        :return: str - "right_sideline" ou "left_sideline" se a bola saiu pela lateral, None caso contrário
        """
        ball = self.frame.ball
        half_wid = self.field.width/2

        if ball.y > half_wid:
            return "UPPER_SIDELINE"
        
        if ball.y < -half_wid:
            return "LOWER_SIDELINE"
        return None
    
    def _check_opponent_defense_area(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        robot_name = f"{color}_{robot.id}"
        half_len = self.field.length/2 
        penalty_x = half_len - self.field.penalty_length
        penalty_y = self.field.penalty_width/2

        if side == "right":
            in_last_third = robot.x < -penalty_x
        else:
            in_last_third = robot.x > penalty_x

        if (
            in_last_third and 
            abs(robot.y) <= penalty_y and
            self.ball_possession == robot_name
        ):
            return "OPPONENT_DEFENSE_AREA"
        return None
    
    def _check_ally_defense_area(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        half_len = self.field.length/2 
        penalty_x = half_len - self.field.penalty_length
        penalty_y = self.field.penalty_width/2
        if side == "right":
            inside_area = robot.x > penalty_x and abs(robot.y) <= penalty_y
        else:
            inside_area = robot.x < -penalty_x and abs(robot.y) <= penalty_y

        n_robots_in_area = getattr(self, f"n_{side}_robots_in_defense")
        if inside_area:
            n_robots_in_area += 1
            setattr(self, f"n_{side}_robots_in_defense", n_robots_in_area)

        if inside_area and n_robots_in_area > 1:
            return "TEAM_DEFENSE_AREA"
        return None
    
    def _check_collision(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        all_robots = {
            **self.frame.robots_blue, 
            **self.frame.robots_yellow
        }

        for other_idx, other_robot in all_robots.items():
            dist = ((robot.x - other_robot.x)**2 + (robot.y - other_robot.y)**2)**(1/2)
            if 0 < dist < 0.25:
                return "COLLISION"
        return None
    
    def _check_double_touch(self):
        if self.is_kickoff == False: return None

        dist = math.hypot(
            self.frame.ball.x - self.init_pos.ball[0], 
            self.frame.ball.y - self.init_pos.ball[1]
        )

        if (
            self.ball_possession is not None and
            dist > 0.1 and 
            len(self.historical_ball_positions) == 1 and
            self.ball_possession in self.historical_ball_positions
        ):
            self.is_kickoff = False
            return "DOUBLE_TOUCH"
        
        elif len(self.historical_ball_positions) == 2:
            self.is_kickoff = False
            return None

        return None


    def _update_offenses(self) -> None:
        self.offenses = {}
        robots_left, robots_right = self.frame.robots_blue, self.frame.robots_yellow
        robot_left_color, robot_right_color = "blue", "yellow"
        if self.left == "yellow":
            robots_left, robots_right = robots_right, robots_left
            robot_left_color, robot_right_color = "yellow", "blue"

        self.n_left_robots_in_defense = 0
        self.n_right_robots_in_defense = 0
        for idx, robot_left in robots_left.items():
            self.offenses[f"{robot_left_color}_{idx}"] = []
            for func in [self._check_opponent_defense_area, self._check_ally_defense_area, self._check_collision]:
                result = func(robot_left, side="left", color=robot_left_color)
                if result:
                    self.offenses[f"{robot_left_color}_{idx}"].append(result)
        
        for idx, robot_right in robots_right.items():
            self.offenses[f"{robot_right_color}_{idx}"] = []
            for func in [self._check_opponent_defense_area, self._check_ally_defense_area,  self._check_collision]:
                result = func(robot_right, side="right", color=robot_right_color)
                if result:
                    self.offenses[f"{robot_right_color}_{idx}"].append(result)
        
        result = self._check_double_touch()
        if result:
            self.offenses[self.ball_possession].append(result)
                  
    def _update_ball_possession(self) -> str|None:
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
        
        ball = self.frame.ball
        n_blue = len(self.frame.robots_blue)
        robots = {
            **self.frame.robots_blue,
            **{idx+n_blue: robot for idx, robot in self.frame.robots_yellow.items()}
        }

        closest_robot = None
        min_distance = float('inf')

        for idx, robot in robots.items():
            distance = math.hypot(ball.x - robot.x, ball.y - robot.y)
            if distance < min_distance:
                min_distance = distance
                closest_robot = robot
                closest_robot.id = idx % n_blue
                closest_robot.yellow = idx // n_blue == 1  # Verifica se é amarelo ou azul

        # Define a zona de domínio do robô como um pouco maior que seu próprio tamanho
        # Isso pode ser ajustado para simular o "controle" da bola
        self.ball_possession = None  
        if not closest_robot: return self.ball_possession
        
        #possession_threshold = closest_robot.rbt_radius * self.possession_radius_scale
        possession_threshold = 0.22 # 0.21 era problematico
        robot_name = f"yellow_{closest_robot.id}" if closest_robot.yellow else f"blue_{closest_robot.id}"
        if min_distance <= possession_threshold:
            self.ball_possession = robot_name
            self.historical_ball_positions.add(self.ball_possession)
        
        return self.ball_possession
              
    def _update_last_touch(self) -> str|None:
        last_ball = self.last_frame.ball
        last_velocity = np.array([last_ball.v_x, last_ball.v_y])
        norm_last_velocity = np.linalg.norm(last_velocity)
        
        ball = self.frame.ball
        current_velocity = np.array([ball.v_x, ball.v_y])
        norm_current_velocity = np.linalg.norm(current_velocity)

        if norm_last_velocity == 0 and norm_current_velocity == 0:
           return self.last_touch

        if norm_last_velocity == 0 and norm_current_velocity > 0:
            last_velocity = -current_velocity 
            norm_last_velocity = norm_current_velocity
        
        if norm_last_velocity > 0 and norm_current_velocity == 0:
            current_velocity = -last_velocity
            norm_current_velocity = norm_last_velocity

        cos_theta = np.dot(last_velocity, current_velocity)
        cos_theta /= (norm_last_velocity * norm_current_velocity)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        direction_changed = angle > self.direction_change_threshold
        if direction_changed and self.ball_possession:
            self.last_touch = self.ball_possession
        
        return self.last_touch
    
    def _get_frame(self, robot_pos: InitialPosition = None):
        
        def random(lim1, lim2): return np.random.uniform(lim1, lim2)

        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2
        positions = []
        frame = Frame()

        frame.ball = Ball(*(robot_pos.ball if robot_pos.ball else [
            random(-field_half_length + 0.1, field_half_length - 0.1), 
            random(-field_half_width + 0.1, field_half_width - 0.1)
        ]))

        min_dist = 0.2
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            is_blue = i < self.n_robots_blue
            idx = i if is_blue else i - self.n_robots_blue
            color = "blue" if is_blue else "yellow"
            pos = getattr(robot_pos, color)[idx] 
            x, y, theta = pos if pos else [
                random(-field_half_length + 0.1, field_half_length - 0.1), 
                random(-field_half_width + 0.1, field_half_width - 0.1), 
                random(0, 360)
            ]
            places = KDTree(positions) if positions else None
            while places is None or places.query([x, y], k=1)[0] < min_dist:
                x, y, theta = (
                    random(-field_half_length + 0.1, field_half_length - 0.1), 
                    random(-field_half_width + 0.1, field_half_width - 0.1), 
                    random(0, 360)
                ) 
                if not places: break
            positions.append([x, y])
            robot_list = getattr(frame, f"robots_{color}")
            robot_list[idx] = Robot(x=x, y=y, theta=theta)
        
        return frame

    def _get_initial_positions_frame(self, stage, ball_pos=None, team_freekick=None):
        '''Returns the position of each robot and ball for the initial frame'''

        if stage == "kickoff":
            frame = self._get_frame(robot_pos=self.init_pos)

            team_last_touch =  ( self.last_touch or "" ).split("_")[0]
            kickoff_team = np.random.choice(["blue", "yellow"])
            if team_last_touch == "yellow":
                kickoff_team = "blue"
            elif team_last_touch == "blue":
                kickoff_team = "yellow"

            robots_list = getattr(frame, f"robots_{kickoff_team}")
            robots_list[0] = Robot(
                x= 0.2 * -(robots_list[0].x / abs(robots_list[0].x)), 
                y= 0.0, 
                theta= robots_list[0].theta + 180.0
            )


        elif stage == "freekick":
            if ball_pos is None: raise ValueError("ball_pos must be provided for freekick")
            if team_freekick not in ["blue", "yellow"]: raise ValueError("team_freekick must be 'blue' or 'yellow'")
            pos = InitialPosition(ball=ball_pos)
            frame = self._get_frame(robot_pos=pos)

            robots_list = getattr(frame, f"robots_{team_freekick}")
            r = 0.2
            f = lambda x:  math.sqrt(r**2 - x**2)
            dx = np.random.uniform(0, r) if team_freekick == "yellow" else np.random.uniform(-r, 0)
            dy = f(dx) if ball_pos[1] > 0 else -f(dx)
            robots_list[0] = Robot(
                x= ball_pos[0] + dx, 
                y= ball_pos[1] + dy, 
                theta= np.random.uniform(0, 360)
            )

        return frame

        

            

            





