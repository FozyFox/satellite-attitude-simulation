import math
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Graph Setup -------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.ion()
plt.show()

def reset_axes():
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

reset_axes()

# Quaternion Math -----------------------------------------

def quaternion_multiply(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2

    return np.array([a, b, c, d])

def quaternion_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_error(q_d, q):
    return quaternion_multiply(q_d, quaternion_inverse(q))

# Physics ------------------------------------

def new_angular_velocity(w, dt, I, T):
    L = I @ w
    L_new = L + T * dt
    return np.linalg.solve(I, L_new)

def update_quaternion(q, w, dt):
    wq = np.array([0, w[0], w[1], w[2]])
    dq = 0.5 * quaternion_multiply(q, wq)
    q_new = q + dq * dt
    q_new /= np.linalg.norm(q_new)
    return q_new

# Controller------------------------------

def new_torque(q_d, q, w):
    K_p = 2.0
    K_d = 7.0

    q_e = quaternion_error(q_d, q)
    if q_e[0] > 0:
        q_e = -q_e

    e = q_e[1:]
    return -K_p * e - K_d * w

# Vector ----------------------------------------

# v_body must be a quaternion with 0 as scalar part
v_body = np.array([0, 1, 0, 0])

def get_sat_display_vector(q):
    # v' = q * v * q^-1
    v_rot = quaternion_multiply(q, quaternion_multiply(v_body, quaternion_inverse(q)))
    return v_rot[1:]

# Initialization ------------------------------------------- (Change values here)

Tx, Ty, Tz = 7.0, 0.7, 0.3
wx, wy, wz = 0.2, 0.2, 0.3

I = np.array([
    [0.2, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.2]
])
# Random Values: NOT ACCURATE

T = np.array([Tx, Ty, Tz])
w = np.array([wx, wy, wz])
q = np.array([1, 0, 0, 0])

# desired quaternion (Enter Desired Quaternion)
q_d = np.array([0.7071, 0, 0, 0.7071])

desired_display_vector = get_sat_display_vector(q_d)

print("Initial Torque:", T)
print("Initial Angular Velocity:", w)
print("Initial Quaternion:", q)
print("Desired Quaternion:", q_d)

# Simulate ----------------------------------

dt = 0.01
steps = 10000

for step in range(steps):

    # physics (yay)
    w = new_angular_velocity(w, dt, I, T)
    q = update_quaternion(q, w, dt)
    T = new_torque(q_d, q, w)

    # display vector
    current_display_vector = get_sat_display_vector(q)

    # redraw every 10 frames
    if step % 10 == 0:
        ax.cla()
        reset_axes()

        ax.quiver(0,0,0,
                  current_display_vector[0],
                  current_display_vector[1],
                  current_display_vector[2],
                  color='r', linewidth=2)

        ax.quiver(0,0,0,
                  desired_display_vector[0],
                  desired_display_vector[1],
                  desired_display_vector[2],
                  color='g', linewidth=2)

        plt.draw()
        plt.pause(0.001)

# other important outputs ------------------------

print()
print("SIMULATION COMPLETE")
print("Simulated time:", steps * dt, "seconds")
print("Final angular velocity:", w)
print("Final quaternion:", q)

# Done :)
