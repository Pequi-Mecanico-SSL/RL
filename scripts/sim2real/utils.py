import numpy as np
from sim2real.config import FIELD_LENGTH, FIELD_WIDTH, NORM_BOUNDS, MAX_POS, MAX_V, MAX_W, ROBOT

def norm_pos(pos):
    return np.clip(
        pos / MAX_POS,
        -NORM_BOUNDS,
        NORM_BOUNDS
    )

def norm_v(v):
    return np.clip(
        v / MAX_V,
        -NORM_BOUNDS,
        NORM_BOUNDS
    )

def norm_w(w):
    return np.clip(
        w / MAX_W,
        -NORM_BOUNDS,
        NORM_BOUNDS
    )

def pos(obj):

    x = norm_pos(obj.x)
    y = norm_pos(obj.y)
    v_x = norm_v(obj.v_x)
    v_y = norm_v(obj.v_y)
    
    theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
    sin = np.sin(theta) if theta is not None else None
    cos = np.cos(theta) if theta is not None else None
    theta = np.arctan2(sin, cos)/np.pi if theta is not None else None
    v_theta = norm_w(obj.v_theta) if theta is not None else None

    return x, y, v_x, v_y, sin, cos, theta, v_theta

def angle_between_three(obj1, obj2, obj3):
    """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

    p1 = np.array([obj1.x, obj1.y])
    p2 = np.array([obj2.x, obj2.y])
    p3 = np.array([obj3.x, obj3.y])

    vec1 = p1 - p2
    vec2 = p3 - p2

    cos_theta = np.arccos(np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    return np.sin(theta), np.cos(theta), theta/np.pi

def angle_between_two(obj1, obj2):
    """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

    p1 = np.array([obj1.x, obj1.y])
    p2 = np.array([obj2.x, obj2.y])

    diff_vec = p1 - p2
    theta = np.arctan2(diff_vec[1], diff_vec[0])

    return np.sin(theta), np.cos(theta), theta/np.pi

def dist_between(obj1, obj2):
    """Retorna a distância formada pela reta que liga o obj1 com obj2"""

    p1 = np.array([obj1.x, obj1.y])
    p2 = np.array([obj2.x, obj2.y])

    diff_vec = p1 - p2
    
    max_dist = np.linalg.norm([FIELD_LENGTH, FIELD_WIDTH])
    dist = np.linalg.norm(diff_vec)

    return np.clip(dist / max_dist, 0, 1)

def inverted_robot(robot):
    return ROBOT(
        -robot.x, 
        robot.y, 
        180 - robot.theta if robot.theta < 180 else 540 - robot.theta, 
        -robot.v_x, 
        -robot.v_y, 
        -robot.v_theta
    )