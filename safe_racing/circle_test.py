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



def test_circles():
    max_steer = 0.4
    radius = L / np.tan(max_steer) 

    pose = [0, 1, np.pi/4]

    dx = radius * np.cos(np.pi/2 -pose[2])
    dy = radius * np.sin(np.pi/2 - pose[2])

    plt.figure()

    c1 = [pose[0] + dx, pose[1] - dy]
    circle = plt.Circle(c1, radius, fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    c2 = [pose[0] - dx, pose[1] + dy]
    circle = plt.Circle(c2, radius, fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    plt.plot(pose[0], pose[1], 'ro')

    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.gca().set_aspect('equal')

    p_l = pose.copy()
    p_r = pose.copy()
    for i in range(50):
        p_l = state_update(p_l, max_steer)
        plt.plot(p_l[0], p_l[1], 'bo')

        p_r = state_update(p_r, -max_steer)
        plt.plot(p_r[0], p_r[1], 'go')

    plt.show()

test_circles()