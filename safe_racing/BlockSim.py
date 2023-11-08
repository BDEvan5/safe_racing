import numpy as np 
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

SPEED = 1
L = 0.33
dT = 0.1

def state_update(state, action):
    x, y, theta = state
    dTheta = SPEED / L * np.tan(action)* dT
    theta = theta + dTheta / 2
    x = x + SPEED * np.cos(theta) * dT
    y = y + SPEED * np.sin(theta)* dT
    theta = theta + dTheta / 2
    return np.array([x, y, theta])


class BlockSim:
    def __init__(self, lefts=[], rights=[]) -> None:
        self.state = np.zeros(3)
        self.history = []
        self.lefts = lefts
        self.rights = rights
        self.N = len(lefts)

    def update(self, action):
        self.state = state_update(self.state, action)
        self.history.append(self.state)

        if self.check_done():
            print(f"{len(self.history)}: Massive Crash Happened!!!!! {self.state}")
            self.history.pop()
            return self.reset()

        return self.state
    
    def check_done(self):
        state_line = [self.history[-2][:2], self.state[:2]]
        for i in range(self.N):
            if do_lines_intersect(self.lefts[i], state_line):
                return True
            elif do_lines_intersect(self.rights[i], state_line):
                return True
        return False
    
    def reset(self):
        self.state = np.zeros(3)
        self.state[0] = 2
        # self.state[2] = 1.5
        self.state[1] = 0.5
        self.history.append(self.state)
        return self.state

    def render(self, axis):
        axis.set_xlim(-1, 6)
        axis.set_ylim(-1, 6)

        history = np.array(self.history)
        axis.plot(history[:, 0], history[:, 1], 'b-')

        axis.plot(self.state[0], self.state[1], 'ro')
        axis.set_aspect('equal')

        for i in range(self.N):
            axis.plot([self.lefts[i][0][0], self.lefts[i][1][0]], [self.lefts[i][0][1], self.lefts[i][1][1]], '--', color='black')
            axis.plot([self.rights[i][0][0], self.rights[i][1][0]], [self.rights[i][0][1], self.rights[i][1][1]], '--', color='black')


def do_lines_intersect(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    # Check if the lines are vertical
    if x1 == x2 and x3 == x4:
        return False
    elif x1 == x2:
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3
        x = x1
        y = m2 * x + b2
    elif x3 == x4:
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        x = x3
        y = m1 * x + b1
    else:
        # Calculate the slopes and y-intercepts of the two lines
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3
        # Check if the lines are parallel
        if m1 == m2:
            return False
        # Calculate the intersection point of the two lines
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    # Check if the intersection point is within the line segments
    if (x < min(x1, x2) or x > max(x1, x2) or
        x < min(x3, x4) or x > max(x3, x4)):
        return False
    if (y < min(y1, y2) or y > max(y1, y2) or
        y < min(y3, y4) or y > max(y3, y4)):
        return False
    return True


#TODO: put the do lines intersect method into the Boundary class to reduce computation
class Boundary:
    def __init__(self, line, extension) -> None:
        self.line = np.array(line, dtype=float)
        self.a = np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
        self.perpendicular = self.a + np.pi/2
        self.norm_vec = np.array([np.cos(self.perpendicular), np.sin(self.perpendicular)])
        self.line[1] += extension * np.array([np.cos(self.a), np.sin(self.a)])
        


class SafetyMask:
    def __init__(self, max_steer, dynamics_fcn, lefts, rights) -> None:
        self.max_steer = max_steer
        self.dynamics_fcn = dynamics_fcn
        self.radius = L / np.tan(max_steer) 
        self.bound_ind = 0
        self.step_counter = -1

        self.N = len(lefts)
        self.left_bounds, self.right_bounds = [], []
        self.polys = []
        for i in range(self.N):
            self.left_bounds.append(Boundary(lefts[i], self.radius))
            self.right_bounds.append(Boundary(rights[i], self.radius))
            poly = Polygon([lefts[i][0], lefts[i][1], rights[i][1], rights[i][0]])
            self.polys.append(poly)

    def enforce_safety(self, state, action, axis):      
        self.step_counter += 1  
        new_state = self.dynamics_fcn(state, action)

        angle_array = self.radius * np.array([-np.cos(np.pi/2 - new_state[2]), np.sin(np.pi/2 - new_state[2])])
        centre_l = new_state[0:2] + angle_array
        centre_r = new_state[0:2] - angle_array

        axis.cla()
        circle = plt.Circle(centre_l, self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)
        circle = plt.Circle(centre_r, self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)

        axis.plot(state[0], state[1], 'ro')
        axis.plot(new_state[0], new_state[1], 'go')

        axis.set_aspect('equal')

        # check what poly I am in
        pt = Point(state[0], state[1])
        if not self.polys[self.bound_ind].contains(pt):
            self.bound_ind += 1
            if self.bound_ind == self.N: self.bound_ind = 0

            if self.polys[self.bound_ind].contains(pt):
                print(f"{self.step_counter}: Switched to next poly: {self.bound_ind-1} --> {self.bound_ind}")
            else:
                print(f"{self.step_counter}: Left previous poly, but not in next..... issue")

        if self.bound_ind == self.N-1:
            search_range = [self.bound_ind, 0]
        else:
            search_range = [self.bound_ind, self.bound_ind+1]
        for i in search_range:
            # i = self.bound_ind + z
            tangent_pt_l = centre_l - self.radius * self.left_bounds[i].norm_vec
            tangent_pt_r = centre_r + self.radius * self.right_bounds[i].norm_vec


            axis.plot([centre_l[0], tangent_pt_l[0]], [centre_l[1], tangent_pt_l[1]], 'g-')
            axis.plot([centre_r[0], tangent_pt_r[0]], [centre_r[1], tangent_pt_r[1]], 'g-')
            axis.plot(tangent_pt_l[0], tangent_pt_l[1], 'go')
            axis.plot(tangent_pt_r[0], tangent_pt_r[1], 'go')

            text_location = [2, 2]
            #TODO: add condition to only check intersection if it is in front of the vehicle.
            radial_line = [centre_l, tangent_pt_l]
            if do_lines_intersect(self.right_bounds[i].line, radial_line):
                print(f"{self.step_counter}: Unsafe - right")
                axis.text(2, 2, "Unsafe - right")
                return self.max_steer

            radial_line = [centre_r, tangent_pt_r]
            if do_lines_intersect(self.left_bounds[i].line, radial_line):
                print(f"{self.step_counter}: Unsafe - left")
                axis.text(2, 2, "Unsafe - left")
                return -self.max_steer

        axis.text(2, 2, "Safe")
        return action


def run_test():
    max_steer = 0.4

    l1_l = [[1, 1], [4, 1]]
    l2_l = [[4, 1], [4, 4]]
    l3_l = [[4, 4], [1, 4]]
    l4_l = [[1, 4], [1, 1]]
    l1_r = [[0, 0], [5, 0]]
    l2_r = [[5, 0], [5, 5]]
    l3_r = [[5, 5], [0, 5]]
    l4_r = [[0, 5], [0, 0]]
    left_lines = [l1_l, l2_l, l3_l, l4_l]
    right_lines = [l1_r, l2_r, l3_r, l4_r]

    sim = BlockSim(left_lines, right_lines)
    supervisor = SafetyMask(max_steer, state_update, left_lines, right_lines)

    actions = []
    safe_actions = []
    state = sim.reset()
    fig = plt.figure(figsize=(9, 9))
    a1 = plt.subplot(1, 1, 1)
    for i in range(350):

        action = np.random.uniform(-max_steer, max_steer)
        # action = max_steer 

        safe_action = supervisor.enforce_safety(state, action, a1)
        actions.append(action)
        safe_actions.append(safe_action)
        state = sim.update(safe_action)

        sim.render(a1)

        plt.tight_layout()
        plt.savefig(f"Data/History_{i}.svg")


run_test()


