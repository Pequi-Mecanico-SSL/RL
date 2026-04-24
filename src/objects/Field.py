from pydantic.dataclasses import dataclass

@dataclass
class Field:
    length: float
    width: float
    penalty_length: float
    penalty_width: float
    goal_width: float
    goal_depth: float