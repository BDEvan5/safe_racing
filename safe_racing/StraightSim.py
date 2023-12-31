import numpy as np 
import matplotlib.pyplot as plt

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


class StraightAngleSim:
    def __init__(self, line1, line2) -> None:
        self.state = np.zeros(3)
        self.history = []
        self.l1s = line1[0]
        self.l1e = line1[1]
        self.l2s = line2[0]
        self.l2e = line2[1]

    def update(self, action):
        self.state = state_update(self.state, action)
        self.history.append(self.state)

        if self.check_done():
            print(f"Massive Crash Happened!!!!! {self.state}")
            self.history.pop()
            return self.reset()

        return self.state
    
    def check_done(self):
        prev_state = self.history[-2][:2]
        if do_lines_intersect([self.l1s, self.l1e], [prev_state, self.state[:2]]):
            return True
        elif do_lines_intersect([self.l2s, self.l2e], [prev_state, self.state[:2]]):
            return True
        return False
    
    def reset(self):
        self.state = np.zeros(3)
        self.state[0] = 0.5
        self.state[2] = 1.5
        # self.state[1] = 0.5
        self.history.append(self.state)
        return self.state

    def render(self, axis):
        axis.set_xlim(-1, 6)
        axis.set_ylim(-1, 6)

        history = np.array(self.history)
        axis.plot(history[:, 0], history[:, 1], 'b-')

        axis.plot(self.state[0], self.state[1], 'ro')
        axis.set_aspect('equal')

        axis.plot([self.l1s[0], self.l1e[0]], [self.l1s[1], self.l1e[1]], '--', color='black')
        axis.plot([self.l2s[0], self.l2e[0]], [self.l2s[1], self.l2e[1]], '--', color='black')

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
    def __init__(self, max_steer, dynamics_fcn, line_l, line_r) -> None:
        self.max_steer = max_steer
        self.dynamics_fcn = dynamics_fcn
        self.radius = L / np.tan(max_steer) 

        self.bound_l = Boundary(line_l, self.radius)
        self.bound_r = Boundary(line_r, self.radius)

    def enforce_safety(self, state, action, axis):        
        new_state = self.dynamics_fcn(state, action)

        angle_array = self.radius * np.array([-np.cos(np.pi/2 - new_state[2]), np.sin(np.pi/2 - new_state[2])])
        centre_l = new_state[0:2] + angle_array
        centre_r = new_state[0:2] - angle_array

        tangent_pt_l = centre_l - self.radius * self.bound_l.norm_vec
        tangent_pt_r = centre_r + self.radius * self.bound_r.norm_vec

        axis.cla()
        circle = plt.Circle(centre_l, self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)
        circle = plt.Circle(centre_r, self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)

        axis.plot(state[0], state[1], 'ro')
        axis.plot(new_state[0], new_state[1], 'go')

        axis.set_aspect('equal')

        axis.plot([centre_l[0], tangent_pt_l[0]], [centre_l[1], tangent_pt_l[1]], 'g-')
        axis.plot([centre_r[0], tangent_pt_r[0]], [centre_r[1], tangent_pt_r[1]], 'g-')
        axis.plot(tangent_pt_l[0], tangent_pt_l[1], 'go')
        axis.plot(tangent_pt_r[0], tangent_pt_r[1], 'go')

        #TODO: add condition to only check intersection if it is in front of the vehicle.
        radial_line = [centre_l, tangent_pt_l]
        if do_lines_intersect(self.bound_r.line, radial_line):
            print("Unsafe - right")
            axis.text(state[0]+2, 0, "Unsafe - right")
            return self.max_steer

        radial_line = [centre_r, tangent_pt_r]
        if do_lines_intersect(self.bound_l.line, radial_line):
            print("Unsafe - left")
            axis.text(state[0]+2, 0, "Unsafe - left")
            return -self.max_steer

        axis.text(state[0]+2, 0, "Safe")
        return action


def run_test():
    max_steer = 0.4
    line_r = [[1, 0], [1, 5]]
    line_l = [[0, 0], [0, 5]]
    # line_r = [[0, 0], [5, 0]]
    # line_l = [[0, 1], [5, 1]]
    # line_l = [[0, 1], [4, 5]]
    # line_r = [[0, 0], [4, 4]]
    sim = StraightAngleSim(line_l, line_r)
    supervisor = SafetyMask(max_steer, state_update, line_l, line_r)

    actions = []
    safe_actions = []
    state = sim.reset()
    fig = plt.figure(figsize=(9, 9))
    a1 = plt.subplot(1, 1, 1)
    # a3 = plt.subplot(2, 1, 2)
    for i in range(80):

        action = np.random.uniform(-max_steer, max_steer)
        # action = max_steer 

        safe_action = supervisor.enforce_safety(state, action, a1)
        actions.append(action)
        safe_actions.append(safe_action)
        state = sim.update(safe_action)

        sim.render(a1)

        # a3.cla()
        # a3.plot(actions, 'b-')
        # a3.plot(safe_actions, 'r-')
        # a3.grid(True)
        # a3.set_xlim(max(0, len(actions)-20), len(actions)+5)

        plt.tight_layout()
        plt.savefig(f"Data/History_{i}.svg")
        # plt.pause(0.5)
        # plt.pause(0.0001)
        # plt.show()


run_test()


