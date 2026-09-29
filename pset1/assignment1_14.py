"""
MAE5810 Homework 1 - Exercise 1.4
"""
import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

# Geometric Parameters
s = 15.0   # hexahedron (platform) side length [cm]
l = 10.0   # pyramid base height [cm]  (z-direction)
w = 15.0   # pyramid base width  [cm]  (y-direction)
h = 20.0   # pyramid length, apex to base [cm] (x-direction)

# Build the platform (hexahedron / cube), centered so bottom sits at z=0
PLATFORM_POS = np.array([0.0, 0.0, s / 2])

def make_box(center, sx, sy, sz):
    """Return the 8 vertices of an axis-aligned box given its center
    and full side lengths (sx, sy, sz)."""
    cx, cy, cz = center
    dx, dy, dz = sx / 2, sy / 2, sz / 2
    verts = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])
    return verts

def box_faces(verts):
    """Return the 6 quad faces of a box given its 8 vertices (order from make_box)."""
    return [
        [verts[0], verts[1], verts[2], verts[3]],  # bottom
        [verts[4], verts[5], verts[6], verts[7]],  # top
        [verts[0], verts[1], verts[5], verts[4]],  # front
        [verts[2], verts[3], verts[7], verts[6]],  # back
        [verts[1], verts[2], verts[6], verts[5]],  # right
        [verts[0], verts[3], verts[7], verts[4]],  # left
    ]

platform_verts = make_box(PLATFORM_POS, s, s, s)
platform_faces = box_faces(platform_verts)


# Build Sensor
# Apex is placed on the +x side of the platform
apex = PLATFORM_POS + np.array([s / 2, 0, 0])

# FOV extends horizontally in the +x direction
base_center = apex + np.array([h, h, 0])

base_verts = np.array([
    [base_center[0], base_center[1] - w / 2, base_center[2] - l / 2],
    [base_center[0], base_center[1] + w / 2, base_center[2] - l / 2],
    [base_center[0], base_center[1] + w / 2, base_center[2] + l / 2],
    [base_center[0], base_center[1] - w / 2, base_center[2] + l / 2],
])

pyramid_faces = [
    [base_verts[0], base_verts[1], base_verts[2], base_verts[3]],
    [apex, base_verts[0], base_verts[1]],
    [apex, base_verts[1], base_verts[2]],
    [apex, base_verts[2], base_verts[3]],
    [apex, base_verts[3], base_verts[0]],
]

# Define one obstacle B and one target T
OBSTACLE_CENTER = np.array([25.0, 20.0, 6.0])
obstacle_dims = (12.0, 12.0, 12.0)

# Target kept inside the horizontal FOV, shifted upward
TARGET_CENTER = np.array([22.0, 0.0, 3.0])
target_dims = (6.0, 6.0, 6.0)

obstacle_verts = make_box(OBSTACLE_CENTER, *obstacle_dims)
obstacle_faces = box_faces(obstacle_verts)

target_verts = make_box(TARGET_CENTER, *target_dims)
target_faces = box_faces(target_verts)


# Intersection Check
def fov_half_extent_at_x(x):
    x_apex = apex[0]
    x_base = base_center[0]

    if x < x_apex or x > x_base:
        return None

    # t = 0 at apex
    # t = 1 at pyramid base
    t = (x - x_apex) / (x_base - x_apex)

    half_w = (w / 2) * t
    half_l = (l / 2) * t

    return half_w, half_l


def box_intersects_fov(center, dims):
    cx, cy, cz = center

    dx = dims[0] / 2
    dy = dims[1] / 2
    dz = dims[2] / 2

    x_min = cx - dx
    x_max = cx + dx

    for x in np.linspace(x_min, x_max, 15):

        extent = fov_half_extent_at_x(x)

        if extent is None:
            continue

        half_w, half_l = extent

        y_overlap = ((cy - dy <= apex[1] + half_w) and (cy + dy >= apex[1] - half_w))

        z_overlap = ((cz - dz <= apex[2] + half_l) and (cz + dz >= apex[2] - half_l))

        if y_overlap and z_overlap:
            return True

    return False


obstacle_hit = box_intersects_fov(OBSTACLE_CENTER, obstacle_dims)
target_hit = box_intersects_fov(TARGET_CENTER, target_dims)

print(f"FOV intersects obstacle B: {obstacle_hit}")
print(f"FOV intersects target  T: {target_hit}")

# Plot objects
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

def add_faces(ax, faces, color, alpha, edgecolor='k', label=None):
    coll = Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor, linewidths=0.8, alpha=alpha)
    ax.add_collection3d(coll)

    if label is not None:
        ax.plot([], [], [], color=color, label=label, linewidth=6)


add_faces(ax, platform_faces, 'steelblue', 0.85, label='Platform A (hexahedron)')
add_faces(ax,pyramid_faces,'orange', 0.25, label='Sensor FOV S (pyramid)')
add_faces(ax, obstacle_faces, 'firebrick', 0.6, label='Obstacle B')
add_faces(ax, target_faces, 'seagreen', 0.9, label='Target T')


# axes limits / labels
ax.set_xlim(-15, 45)
ax.set_ylim(-30, 35)
ax.set_zlim(0, 30)

ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')

ax.set_title('Exercise 1.4: Robotic Sensor (Platform + FOV) with Obstacle and Target')

ax.legend(loc='upper left')
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.savefig('exercise_1_4_plot.png', dpi=200)

print("Saved plot to exercise_1_4_plot.png")

plt.show()