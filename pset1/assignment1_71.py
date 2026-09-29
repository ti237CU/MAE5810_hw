"""
Exercise 2.1(c) - Truck Backer-Upper Simulation
"""

import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt


# Geometric parameters
L = 5.0   # cabin wheelbase [m]
d = 8.0   # hitch-to-trailer-axle distance [m]

dt = 0.05     # integration time step [s]


# Kinematics
def truck_kinematics(state, u, L, d):
    x_c, y_c, theta_c, theta_t = state
    v_c, phi = u

    xdot_c = v_c * np.cos(theta_c)
    ydot_c = v_c * np.sin(theta_c)
    thetadot_c = (v_c / L) * np.tan(phi - theta_c)
    thetadot_t = (v_c / d) * np.sin(theta_c - theta_t)

    return np.array([xdot_c, ydot_c, thetadot_c, thetadot_t])

# Runga Kutta 4th order
def rk4_step(state, u, dt, L, d):
    k1 = truck_kinematics(state, u, L, d)
    k2 = truck_kinematics(state + 0.5 * dt * k1, u, L, d)
    k3 = truck_kinematics(state + 0.5 * dt * k2, u, L, d)
    k4 = truck_kinematics(state + dt * k3, u, L, d)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Compute trailer position as a function of cabin
def trailer_axle_position(state, d):
    x_c, y_c, theta_c, theta_t = state
    x_t = x_c - d * np.cos(theta_t)
    y_t = y_c - d * np.sin(theta_t)
    return x_t, y_t


# Input dynamics: drive left, drive forward, then reverse left
def input_schedule(t, state):
    theta_c = state[2]

    if t < 8:
        v_c = 2.0
        delta = np.deg2rad(5)
        phi = theta_c + delta

    elif t < 25:
        v_c = 2.0
        phi = theta_c

    elif t < 35:
        v_c = -2.0
        delta = np.deg2rad(-3) * np.sin(0.5 * (t - 25))
        phi = theta_c + delta

    else:
        v_c = -2.0
        phi = theta_c

    return v_c, phi


# Simulate
t_final = 40.0
n_steps = int(t_final / dt)

state = np.array([0.0, 0.0, np.deg2rad(0), np.deg2rad(0)])

time_hist = np.zeros(n_steps + 1)
cabin_hist = np.zeros((n_steps + 1, 2))
trailer_hist = np.zeros((n_steps + 1, 2))
theta_c_hist = np.zeros(n_steps + 1)
theta_t_hist = np.zeros(n_steps + 1)
phi_hist = np.zeros(n_steps + 1)

cabin_hist[0] = state[0:2]
trailer_hist[0] = trailer_axle_position(state, d)
theta_c_hist[0] = state[2]
theta_t_hist[0] = state[3]

for i in range(n_steps):
    t = i * dt
    u = input_schedule(t, state)
    phi_hist[i] = u[1]

    state = rk4_step(state, u, dt, L, d)

    time_hist[i + 1] = t + dt
    cabin_hist[i + 1] = state[0:2]
    trailer_hist[i + 1] = trailer_axle_position(state, d)
    theta_c_hist[i + 1] = state[2]
    theta_t_hist[i + 1] = state[3]

phi_hist[-1] = phi_hist[-2]


# Sketch out simulation
def rect_outline(anchor_x, anchor_y, theta, length, width, anchor_is_front):

    local = np.array([
        [0.0,  width / 2],
        [0.0, -width / 2],
        [length, -width / 2],
        [length,  width / 2],
        [0.0,  width / 2],  
    ])

    if anchor_is_front:
        local[:, 0] -= length

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    world = (R @ local.T).T + np.array([anchor_x, anchor_y])
    return world



cabin_length = 5.0
cabin_width = 5.0
trailer_length = d
trailer_width = 2.5

fig, ax = plt.subplots(figsize=(10, 9))

# full path traces
ax.plot(cabin_hist[:, 0], cabin_hist[:, 1], 'b--', linewidth=1, alpha=0.5,
        label='Cabin hitch-point path')
ax.plot(trailer_hist[:, 0], trailer_hist[:, 1], 'r--', linewidth=1, alpha=0.5,
        label='Trailer axle path')

# draw truck+trailer outlines at several snapshots in time
n_snap = 10
snap_idx = np.linspace(0, n_steps, n_snap, dtype=int)
for k, idx in enumerate(snap_idx):
    xc, yc = cabin_hist[idx]
    th_c = theta_c_hist[idx]
    th_t = theta_t_hist[idx]
    alpha = 0.35 + 0.65 * (k / (n_snap - 1))
    
    cabin_poly = rect_outline(xc, yc, th_c, cabin_length, cabin_width,
                               anchor_is_front=False)
    trailer_poly = rect_outline(xc, yc, th_t, trailer_length, trailer_width,
                                 anchor_is_front=True)

    ax.fill(trailer_poly[:, 0], trailer_poly[:, 1],
            facecolor='none', edgecolor='red', linewidth=1.5, alpha=alpha)
    ax.fill(cabin_poly[:, 0], cabin_poly[:, 1],
            facecolor='gold', edgecolor='black', linewidth=1.5, alpha=alpha)

    # hitch marker
    ax.plot(xc, yc, 'k.', markersize=3)

ax.plot(cabin_hist[0, 0], cabin_hist[0, 1], 'go', markersize=8, label='Start')
ax.plot(cabin_hist[-1, 0], cabin_hist[-1, 1], 'ks', markersize=8, label='End')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Truck Backer-Upper: Cabin and Trailer Outlines Along the Path')
ax.legend(loc='best')
ax.axis('equal')
ax.grid(True)

plt.tight_layout()
plt.savefig('truck_backer_upper_outline.png', dpi=150)
plt.show()

print("Simulation complete.")
print(f"Final cabin state:   x={cabin_hist[-1,0]:.2f}, y={cabin_hist[-1,1]:.2f}, "
      f"theta_c={np.degrees(theta_c_hist[-1]):.1f} deg")
print(f"Final trailer state: x={trailer_hist[-1,0]:.2f}, y={trailer_hist[-1,1]:.2f}, "
      f"theta_t={np.degrees(theta_t_hist[-1]):.1f} deg")