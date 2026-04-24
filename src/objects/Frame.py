from typing import Dict
from src.objects.Ball import Ball
from src.objects.Robot import Robot
from pydantic.dataclasses import dataclass
from dataclasses import field
@dataclass
class Frame:
    ball: Ball = Ball()
    robots_blue: Dict[int, Robot] = field(default_factory=dict)
    robots_yellow: Dict[int, Robot] = field(default_factory=dict)