from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class InitialPosition(BaseModel):
    blue: Dict[int, Optional[List[float]]] = Field(default_factory=dict)
    yellow: Dict[int, Optional[List[float]]] = Field(default_factory=dict)
    ball: Optional[List[float]] = None

    @field_validator("blue", "yellow", mode="before")
    @classmethod
    def _normalize_team_positions(cls, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            normalized = {}
            for key, pos in value.items():
                if isinstance(pos, str) and pos.strip().lower() in {"none", "null", ""}:
                    normalized[key] = None
                else:
                    normalized[key] = pos
            return normalized
        return value

    @field_validator("ball", mode="before")
    @classmethod
    def _normalize_ball(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return value


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