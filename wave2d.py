import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# 1. Parameters
# -------------------------
length = 10.0
c = 1.0                 # wave speed
dx = 0.05
dy = dx
dt = dx / (c * np.sqrt(2)) * 0.9
total_time = 10.0  # shorter for animation speed
nt = int(total_time / dt)

nx = int(length / dx) + 1
ny = int(length / dy) + 1

# -------------------------
# 2. Grid coordinates
# -------------------------
x = np.linspace(0, length, nx)
y = np.linspace(0, length, ny)
X, Y = np.meshgrid(x, y)

# -------------------------
# 3. Initial displacement (Gaussian)
# -------------------------
x0s = (length/3, length/3*2.5)
y0s = (length/3, length/3*2.5)
sigma = 0.5

phi = np.sum([np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma**2) for x0, y0 in zip(x0s, y0s)], axis=0)
psi = np.zeros((nx, ny))

# Allocate solution arrays
u_prev = phi.copy()      # u^0
u = np.zeros((nx, ny))   # u^1
u_next = np.zeros((nx, ny))

# Compute first time step
u[1:-1, 1:-1] = (
    u_prev[1:-1, 1:-1]
    + dt * psi[1:-1, 1:-1]
    + 0.5 * (c * dt)**2 * (
        (u_prev[2:, 1:-1] - 2*u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]) / dx**2
        +
        (u_prev[1:-1, 2:] - 2*u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]) / dy**2
    )
)
# Dirichlet boundaries
u[0, :] = u[-1, :] = 0
u[:, 0] = u[:, -1] = 0

# -------------------------
# 4. Animation setup
# -------------------------
fig, ax = plt.subplots()
im = ax.imshow(u, extent=[0, length, 0, length], origin='lower',
               vmin=-1, vmax=1, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D Wave Propagation')

def update(frame):
    global u_prev, u, u_next
    # Compute next time step
    u_next[1:-1, 1:-1] = (
        2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
        + (c * dt)**2 * (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
    )
    # Dirichlet boundaries
    u_next[0, :] = u_next[-1, :] = 0
    u_next[:, 0] = u_next[:, -1] = 0

    # Update arrays for next iteration
    u_prev, u, u_next = u, u_next, u_prev

    # Update plot
    im.set_array(u)
    return [im]

ani = FuncAnimation(fig, update, frames=nt, interval=30, blit=True)
plt.show()