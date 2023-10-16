import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetBox, AnnotationBbox, OffsetImage



def rotate_bound(image, angle):
    import cv2
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=(1, 1, 1))


def plot_imgs():
    fig = plt.figure(figsize=(8, 2))
    a1 = plt.subplot(131)
    a2 = plt.subplot(132, sharey=a1)
    a3 = plt.subplot(133, sharey=a1)

    for a in (a1, a2, a3):
        road = Rectangle((0, -1), 5, 2, facecolor='#C0C0C0')
        a.add_patch(road)
        a.set_xlim(0.6, 3.5)
        a.set_ylim(-0.5, 1.5)
        a.set_xticks([])
        a.set_yticks([])

        a.plot([0, 5], [0, 0], '--', color="orange", linewidth=2.5)
        a.plot([0, 5], [1, 1], '-', color="black", linewidth=1.5)
        a.set_aspect('equal', adjustable='box')

    a1.set_title("Safe")
    a2.set_title("Unsafe")

    max_steer = 0.3

    angle = 0.7
    img_raw = plt.imread("RacingCar.png", format='png')
    img = rotate_bound(img_raw, 90-np.rad2deg(angle))
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.3, 0.22), xycoords='data', frameon=False)
    a1.add_artist(ab)

    init_state = np.array([1.5, 0.4, angle])
    left_poses, right_poses = make_cricles(init_state, max_steer)

    xs, ys, thetas = zip(*left_poses)
    a1.plot(xs, ys, color='red', linewidth=2.5)

    xs, ys, thetas = zip(*right_poses)
    a1.plot(xs, ys, color='red', linewidth=2.5)


    angle = 70
    img = rotate_bound(img_raw, 90-angle)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.4, 0.15), xycoords='data', frameon=False)
    a2.add_artist(ab)

    init_state = np.array([1.5, 0.38, np.deg2rad(angle)])
    left_poses, right_poses = make_cricles(init_state, max_steer)

    xs, ys, thetas = zip(*left_poses)
    a2.plot(xs, ys, color='red', linewidth=2.5)

    xs, ys, thetas = zip(*right_poses)
    a2.plot(xs, ys, color='red', linewidth=2.5)




    angle = 1.03
    img = rotate_bound(img_raw, 90-np.rad2deg(angle))
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.35, 0.2), xycoords='data', frameon=False)
    a3.add_artist(ab)

    init_state = np.array([1.5, 0.4, angle])
    left_poses, right_poses = make_cricles(init_state, max_steer)

    xs, ys, thetas = zip(*left_poses)
    a3.plot(xs, ys, color='red', linewidth=2.5)

    xs, ys, thetas = zip(*right_poses)
    a3.plot(xs, ys, color='red', linewidth=2.5)

    a3.set_title("Marginally safe")

    plt.tight_layout()
    plt.savefig("Imgs/tangent_circles.svg")
    plt.savefig("Imgs/tangent_circles.pdf")



def make_cricles(init_pose, max_steer):
    pose_left = init_pose.copy()
    pose_right = init_pose.copy()

    left_poses, right_poses = [], []
    for i in range(50):
        pose_left = state_update(pose_left, max_steer)
        pose_right = state_update(pose_right, -max_steer)
        left_poses.append(pose_left)
        right_poses.append(pose_right)
    
    return left_poses, right_poses


SPEED = 1
L = 0.33
t = 0.05
def state_update(state, action):
    x, y, theta = state
    x = x + SPEED * np.cos(theta) * t
    y = y + SPEED * np.sin(theta)* t
    theta = theta + SPEED / L * np.tan(action)* t
    return np.array([x, y, theta])


plot_imgs()