from typing import Dict, List
from dataclasses import field
from pydantic import BaseModel

class InitialPosition(BaseModel):
    blue: Dict[int, List[float]] = field(default_factory=dict)
    yellow: Dict[int, List[float]] = field(default_factory=dict)
    ball: List[float] = field(default_factory=dict)


class Config(BaseModel):
    init_pos: InitialPosition
    field_type: int = 1
    stack_size: int = 8
    fps: int = 30
    match_time: int = 40
    processesion_radius_scale: float = 3
    direction_change_threshold: float = 1
    render_mode: str = 'human'

if __name__ == "__main__":
    config = Config(
        init_pos=InitialPosition(
            blue={
                0: [0, -1, 0],
                1: [-0.5, -1, 0],
                2: [0.5, -1, 0]
            },
            yellow={
                0: [0, 1, 3.14],
                1: [-0.5, 1, 3.14],
                2: [0.5, 1, 3.14]
            },
            ball=[0, 0]
        )
    )
    print(config.dict())