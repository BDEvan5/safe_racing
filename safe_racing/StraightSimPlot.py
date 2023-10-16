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


class StraightSim:
    def __init__(self, track_width) -> None:
        self.state = np.zeros(3)
        self.history = [self.state]
        self.track_width = track_width

    def update(self, action):
        self.state = state_update(self.state, action)
        self.history.append(self.state)

        if self.state[1] > self.track_width or self.state[1] < - self.track_width:
            print(f"Massive Crash Happened!!!!!")
            self.history.pop()
            return self.reset()

        return self.state
    
    
    def reset(self):
        self.state = np.zeros(3)
        return self.state

    def render(self, axis):
        axis.cla()
        axis.set_xlim(self.state[0]-4, self.state[0]+4)
        axis.set_ylim(-self.track_width*1.2, self.track_width*1.2)

        history = np.array(self.history)
        axis.plot(history[:, 0], history[:, 1], 'b-')

        axis.plot(self.state[0], self.state[1], 'ro')
        axis.set_aspect('equal')

        axis.plot([0, 100], [self.track_width, self.track_width], '--', color='black')
        axis.plot([0, 100], [-self.track_width, -self.track_width], '--', color='black')


class SafetyMask:
    def __init__(self, max_steer, dynamics_fcn, track_width) -> None:
        self.max_steer = max_steer
        self.dynamics_fcn = dynamics_fcn
        self.track_width = track_width *0.9

        self.radius = L / np.tan(max_steer) 

    def plot_extra_circles(self, ns, axis):
        cx_r = ns[0] + self.radius * np.cos(np.pi/2 - ns[2])
        cy_r = ns[1] - self.radius * np.sin(np.pi/2 - ns[2])
        cx_l = ns[0] - self.radius * np.cos(np.pi/2 - ns[2])
        cy_l = ns[1] + self.radius * np.sin(np.pi/2 - ns[2])

        circle = plt.Circle((cx_r, cy_r), self.radius, fill=False, linewidth=2, color='green')
        axis.add_artist(circle)
        circle = plt.Circle((cx_l, cy_l), self.radius, fill=False, linewidth=2, color='green')
        axis.add_artist(circle)


    def enforce_safety(self, state, action, axis):        
        new_state = self.dynamics_fcn(state, action)

        cx_r = new_state[0] + self.radius * np.cos(np.pi/2 - new_state[2])
        cy_r = new_state[1] - self.radius * np.sin(np.pi/2 - new_state[2])
        cx_l = new_state[0] - self.radius * np.cos(np.pi/2 - new_state[2])
        cy_l = new_state[1] + self.radius * np.sin(np.pi/2 - new_state[2])

        max_y_r = cy_r + self.radius 
        min_y_l = cy_l - self.radius

        axis.cla()
        axis.set_xlim(state[0]-1.5, state[0]+2)
        axis.set_ylim(-self.track_width*1.2, self.track_width*1.2)

        circle = plt.Circle((cx_r, cy_r), self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)
        circle = plt.Circle((cx_l, cy_l), self.radius, fill=False, linewidth=2)
        axis.add_artist(circle)

        axis.plot(state[0], state[1], 'ro')
        axis.plot(new_state[0], new_state[1], 'go')

        axis.plot([0, 100], [self.track_width, self.track_width], '--', color='black')
        axis.plot([0, 100], [-self.track_width, -self.track_width], '--', color='black')
        axis.set_aspect('equal')

        if max_y_r > self.track_width and cx_r > new_state[0] and new_state[1] > 0:
            print("Correcting unsafe action - LHS")
            axis.text(state[0]-1, 0, "Unsafe - LHS")
            return -self.max_steer 
        elif min_y_l < -self.track_width and cx_l > new_state[0] and new_state[1] < 0:
            print("Correcting unsafe action - RHS")
            axis.text(state[0]-1, 0, "Unsafe - RHS")
            return self.max_steer
        else:
            axis.text(state[0]-1, 0, "Safe")
            return action


def run_test():
    track_width = 0.4 # m on each side of 0 center
    max_steer = 0.4
    sim = StraightSim(track_width)
    supervisor = SafetyMask(max_steer, state_update, track_width)

    actions = []
    safe_actions = []
    state = sim.reset()
    fig = plt.figure(figsize=(9, 6))
    a1 = plt.subplot(3, 1, 1)
    a2 = plt.subplot(3, 1, 2)
    a3 = plt.subplot(3, 1, 3)
    for i in range(300):
        action = np.random.uniform(-max_steer, max_steer)

        safe_action = supervisor.enforce_safety(state, action, a2)
        actions.append(action)
        safe_actions.append(safe_action)
        state = sim.update(safe_action)
        if state[1] > track_width*0.9 or state[1] < -track_width*0.9:
            print(f"Crashing....")

        sim.render(a1)

        a3.cla()
        a3.plot(actions, 'b-', label="Original action")
        a3.plot(safe_actions, 'r-', label="Safe action")
        a3.grid(True)
        a3.legend()
        a3.set_xlim(max(0, len(actions)-30), len(actions)+5)
        a3.set_ylabel("Steering Action")

        plt.tight_layout()
        plt.savefig(f"Data/History_{i}.svg")
        # plt.pause(0.5)
        # plt.pause(0.0001)
        # plt.show()


run_test()


