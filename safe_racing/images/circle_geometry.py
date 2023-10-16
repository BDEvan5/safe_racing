import matplotlib.pyplot as plt
import numpy as np

def circle_center(radius, point, angle):
    x, y = point
    dx = radius * np.cos(np.pi/2 -angle)
    dy = radius * np.sin(np.pi/2 - angle)

    return x + dx, y - dy

# example data
radius = 0.8
point = (1, 0.4)
tangent_angle = 0.5

# calculate center point
center = circle_center(radius, point, tangent_angle)
print(f"Center: {center} points")

# plot circle, point, and tangent line
# fig, ax = plt.subplots()
fig = plt.figure(figsize=(5, 2.5))
ax = plt.gca()
circle = plt.Circle(center, radius, fill=False, linewidth=2)
ax.add_artist(circle)
plt.plot(center[0], center[1], 'o', color='blue', markersize=12)

plt.plot(point[0], point[1], 'o', color='red', markersize=10)
length_a = 0.1
plt.arrow(point[0], point[1], np.cos(tangent_angle)*length_a, length_a*np.sin(tangent_angle), head_width=0.04, head_length=0.04, linewidth=2, fc='red', ec='red')

plt.plot([point[0], center[0]], [point[1], center[1]], color='black', linestyle='--', linewidth=2)
# plt.plot([point[0], point[0] + 0.5], [point[1], point[1]], color='black', linestyle='--')

plt.plot(center[0], center[1] + radius, 'o', color='green', markersize=10)
plt.text(center[0], center[1] + radius*0.85, "Max y", ha='center', va='bottom')
plt.text(center[0]+0.14, center[1] , "Centre \npoint", ha='center', va='bottom')
plt.text(0.8, 0.4, "Vehicle", ha='center', va='bottom')
plt.text(center[0]-0.1, center[1] + radius/2, "Radius", ha='center', va='bottom')

plt.xlim(0.5, 2)

ax.set_aspect('equal')
plt.xticks([])
plt.yticks([])

plt.tight_layout()

plt.savefig(f"Imgs/CircleGeometry.svg", pad_inches=0.02, bbox_inches='tight')
plt.savefig(f"Imgs/CircleGeometry.pdf", pad_inches=0.02, bbox_inches='tight')


