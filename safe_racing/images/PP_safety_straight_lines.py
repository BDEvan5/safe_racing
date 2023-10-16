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
        a.set_xlim(0, 5)
        a.set_ylim(-1, 1)
        a.set_xticks([])
        a.set_yticks([])

        a.plot([0, 5], [0, 0], '--', color="orange", linewidth=2.5)

    a1.set_title("Safe")
    a2.set_title("Unsafe")
    a3.set_title("????????")

    img_raw = plt.imread("RacingCar.png", format='png')
    img = rotate_bound(img_raw, 90)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, -0.4), xycoords='data', frameon=False)
    a1.add_artist(ab)

    img = rotate_bound(img_raw, 25)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, 0.4), xycoords='data', frameon=False)
    a2.add_artist(ab)

    img = rotate_bound(img_raw, 50)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (1.5, 0.2), xycoords='data', frameon=False)
    a3.add_artist(ab)

    plt.tight_layout()
    plt.savefig("Imgs/pp_safety_straight.svg")
    plt.savefig("Imgs/pp_safety_straight.pdf")



plot_imgs()