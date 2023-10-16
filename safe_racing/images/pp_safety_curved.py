import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetBox, AnnotationBbox, OffsetImage
from scipy.interpolate import splprep, splev

def plot_imgs():
    fig = plt.figure(figsize=(8, 2.2))
    a1 = plt.subplot(121)
    a2 = plt.subplot(122)

    xs = np.linspace(0, 7, 100)
    ys = np.sin(xs)
    a1.plot(xs, ys, color='grey', linewidth=45)
    a1.plot(xs, ys, '--', color='orange', linewidth=2)
    a1.set_xticks([])
    a1.set_yticks([])

    a1.set_ylim(-1.8, 1.8)

    t = np.linspace(0, 1, 1000)
    tck, _ = splprep([xs, ys], k=3, s=100)
    dx, dy = splev(t, tck, der=1)
    d2x, d2y = splev(t, tck, der=2)
    curvature2 = (dx * d2y  - dy * d2x) / (dx**2 + dy **2) ** 1.5

    a2.plot(t*7, curvature2, linewidth=3, color='red')
    a2.grid(True)
    # a2.legend(["Curvature"])
    a2.set_xlabel("Centre line distance")
    a2.set_ylabel("Curvature")

    plt.tight_layout()
    plt.savefig("Imgs/pp_safety_curved.svg")
    plt.savefig("Imgs/pp_safety_curved.pdf")



plot_imgs()