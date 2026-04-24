import numpy as np

class Geometry2D():
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax    

    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])
        p3 = np.array([obj3['x'], obj3['y']])

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.xmax - self.xmin, self.ymax - self.ymin])
        dist = np.linalg.norm(diff_vec)

        return np.clip(dist / max_dist, 0, 1)
    
    def _invert_coordinates(self, obj, on_x = False, on_y = False):
        if on_x:
            obj['x'] = -obj['x']
            obj['theta'] = 180 - obj['theta'] if obj['theta'] < 180 else 540 - obj['theta']
        
        if on_y:
            obj['y'] = -obj['y']
            obj['theta'] = -obj['theta']

        return obj
