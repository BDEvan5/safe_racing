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
        road = Rectangle((0, -1), 5, 2, facecolor='gray')
        a.add_patch(road)
        a.set_xlim(0.6, 3.5)
        a.set_ylim(-1, 1)
        a.set_xticks([])
        a.set_yticks([])

        a.plot([0, 5], [0, 0], '--', color="orange", linewidth=2.5)

    a1.set_title("Safe")
    a2.set_title("Unsafe")

    img_raw = plt.imread("RacingCar.png", format='png')
    img = rotate_bound(img_raw, 90)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, -0.4), xycoords='data', frameon=False)
    a1.add_artist(ab)

    ts = np.linspace(0, 2, 100)
    init_state = (1.5, -0.4, 0)
    action = 0.4
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a1.plot(xs, ys, color='red', linewidth=2.5)

    init_state = (1.5, -0.4, 0)
    action = -0.40
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a1.plot(xs, ys, color='red', linewidth=2.5)


    angle = 78
    img = rotate_bound(img_raw, 90-angle)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, 0.4), xycoords='data', frameon=False)
    a2.add_artist(ab)

    ts = np.linspace(0, 2, 100)
    init_state = (1.5, 0.4, np.deg2rad(angle))
    action = 0.4
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a2.plot(xs, ys, color='red', linewidth=2.5)

    action = -0.40
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a2.plot(xs, ys, color='red', linewidth=2.5)



    angle = 1.15
    img = rotate_bound(img_raw, 90-np.rad2deg(angle))
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, 0.4), xycoords='data', frameon=False)
    a3.add_artist(ab)

    ts = np.linspace(0, 2, 100)
    init_state = (1.5, 0.4, angle)
    action = 0.4
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a3.plot(xs, ys, color='red', linewidth=2.5)

    action = -0.40
    states = [init_state]
    for t in ts:
        states.append(state_update(states[-1], action))

    xs, ys, thetas = zip(*states)
    a3.plot(xs, ys, color='red', linewidth=2.5)
    a3.set_title("Marginally safe")

    plt.tight_layout()
    plt.savefig("Imgs/pp_safety_derivation.svg")
    plt.savefig("Imgs/pp_safety_derivation.pdf")

SPEED = 1
L = 0.33
t = 0.02
def state_update(state, action):
    x, y, theta = state
    x = x + SPEED * np.cos(theta) * t
    y = y + SPEED * np.sin(theta)* t
    theta = theta + SPEED / L * np.tan(action)* t
    return np.array([x, y, theta])


plot_imgs()