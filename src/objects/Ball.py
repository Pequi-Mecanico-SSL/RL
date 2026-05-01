from pydantic.dataclasses import dataclass

@dataclass
class Ball:
    x: float = None
    y: float = None
    v_x: float = 0
    v_y: float = 0