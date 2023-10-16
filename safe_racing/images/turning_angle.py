import numpy as np
import matplotlib.pyplot as plt

SPEED = 1
L = 0.33
dt = 0.2

def dynamics(pose, action):
    pose[0] += SPEED * np.cos(pose[2]) * dt
    pose[1] += SPEED * np.sin(pose[2]) * dt
    pose[2] += np.tan(action) * SPEED / L  * dt
    return pose


def turning_angle():
    fig = plt.figure(figsize=(5, 3))


    max_steer = 0.4
    a_length = 0.08
    a_length2 = 0.1
    pose = [0, 1, np.pi/2]

    plt.plot(pose[0], pose[1], 'o', color='#eb3b5a')
    # plt.arrow(pose[0], pose[1], 0.5 * np.cos(pose[2]), 0.5 * np.sin(pose[2]), head_width=0.1, head_length=0.1)
    plt.arrow(pose[0], pose[1], a_length2 * np.cos(pose[2]), a_length2 * np.sin(pose[2]), head_width=0.04, head_length=0.05, fc="#eb3b5a", ec="#eb3b5a")
    pose_left = pose.copy()
    pose_right = pose.copy()
    for i in range(20):
        pose_left = dynamics(pose_left, max_steer)
        plt.plot(pose_left[0], pose_left[1], '.', color='#3867d6')
        plt.arrow(pose_left[0], pose_left[1], a_length * np.cos(pose_left[2]), a_length * np.sin(pose_left[2]), head_width=0.03, head_length=0.04, fc='#4b7bec', ec="#4b7bec")

        pose_right = dynamics(pose_right, -max_steer)
        plt.arrow(pose_right[0], pose_right[1], a_length * np.cos(pose_right[2]), a_length * np.sin(pose_right[2]), head_width=0.03, head_length=0.04, fc="#a55eea", ec="#a55eea")
        plt.plot(pose_right[0], pose_right[1], '.', color='#8854d0')

    plt.xlim(-2, 2)
    plt.ylim(0.1, 2.2)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.text(0.8, 1.1, "Right turning \ncircle", ha='center')
    plt.text(-0.8, 1.1, "Left turning \ncircle", ha='center')
    plt.text(0, 1.87, "Driveable \narea", ha='center')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.savefig(f"Imgs/tuning_angle.svg", pad_inches=0.0, bbox_inches='tight')
    plt.savefig(f"Imgs/tuning_angle.pdf", pad_inches=0.0, bbox_inches='tight')

turning_angle()
