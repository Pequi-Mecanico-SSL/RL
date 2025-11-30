import pygame
from utils import COLORS
import math


class Ball:
    radius = 0.04

    def __init__(self, x, y, scale) -> None:
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.pvx = 0  # Adicionado para a velocidade x anterior
        self.pvy = 0  # Adicionado para a velocidade y anterior
        self.friction = 0.98
        self.radius *= scale
        self.scale = scale

    def update(self):
        self.pvx = self.vx # Adicionado
        self.pvy = self.vy # Adicionado

        self.x += self.vx
        self.y += self.vy

        self.vx *= self.friction
        self.vy *= self.friction

        if abs(self.vx) < 0.01:
            self.vx = 0
        if abs(self.vy) < 0.01:
            self.vy = 0

    def kick(self, force_x, force_y):
        self.vx += force_x
        self.vy += force_y


    def draw(self, screen):
        pygame.draw.circle(screen, COLORS["ORANGE"], (self.x, self.y), self.radius)
        pygame.draw.circle(screen, COLORS["BLACK"], (self.x, self.y), self.radius, 1)
        
    def check_collision(self, robo):
        dist = math.hypot(self.x - robo.x, self.y - robo.y)
        return dist < self.radius + robo.size

    def move_limited(self, dx, dy, bounds):
        new_x = self.x + dx
        new_y = self.y + dy

        min_x, max_x, min_y, max_y = bounds
        self.x = max(min_x, min(max_x, new_x))
        self.y = max(min_y, min(max_y, new_y))

    def bounce_off(self, robo):
        dx = self.x - robo.x
        dy = self.y - robo.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return None
        norm_dx = dx / dist
        norm_dy = dy / dist
        force = 2.5
        self.kick(norm_dx * force, norm_dy * force)
        return robo




