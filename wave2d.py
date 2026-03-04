import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# 1. Parameters
# -------------------------
length = 10.0
c = 1.0
dx = 0.05
dy = dx
dt = dx / (c * np.sqrt(2)) * 0.9
total_time = 10.0
nt = int(total_time / dt)

nx = int(length / dx) + 1
ny = int(length / dy) + 1

# Stencil coefficients (nearest neighbor example)
coef = np.array([-2, 1])  # can replace with higher-order stencil
max_r = coef.size - 1

# -------------------------
# 2. Grid coordinates
# -------------------------
x = np.linspace(0, length, nx)
y = np.linspace(0, length, ny)
X, Y = np.meshgrid(x, y)

# -------------------------
# 3. Multiple sources
# -------------------------
sources = [
    (length / 3, length / 3, 0.5, 1.0),
    (length / 3 * 2.5, length / 3 * 2.5, 0.5, 1.0)
]

phi = np.zeros((nx, ny))
for x0, y0, sigma, A in sources:
    phi += A * np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma**2)

psi = np.zeros((nx, ny))

# -------------------------
# 4. Solution arrays
# -------------------------
u_prev = phi.copy()  # u^0
u = np.zeros((nx, ny))  # u^1
u_next = np.zeros((nx, ny))

dx2 = dx**2
dy2 = dy**2
dt2c2 = (c * dt)**2

# First time step
laplacian = (
    (np.roll(u_prev, -1, axis=0) - 2 * u_prev + np.roll(u_prev, 1, axis=0)) / dx2 +
    (np.roll(u_prev, -1, axis=1) - 2 * u_prev + np.roll(u_prev, 1, axis=1)) / dy2
)
u[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * psi[1:-1, 1:-1] + 0.5 * dt2c2 * laplacian[1:-1, 1:-1]

# Dirichlet boundaries
u[0, :] = u[-1, :] = 0
u[:, 0] = u[:, -1] = 0

# -------------------------
# 5. Animation setup
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


    # Pad u with 2 layers of zeros for 5-point stencil (Dirichlet)
    u_next[2:-2, 2:-2] = (
        2*u[2:-2, 2:-2] - u_prev[2:-2, 2:-2]
        + (c * dt)**2 / dx**2 / 12 * (
            (-u[4:, 2:-2] + 16*u[3:-1, 2:-2] - 30*u[2:-2, 2:-2] + 16*u[1:-3, 2:-2] - u[:-4, 2:-2])
            +
            (-u[2:-2, 4:] + 16*u[2:-2, 3:-1] - 30*u[2:-2, 2:-2] + 16*u[2:-2, 1:-3] - u[2:-2, :-4])
        )
    )

    # Swap arrays
    u_prev, u, u_next = u, u_next, u_prev

    im.set_array(u)
    return [im]

ani = FuncAnimation(fig, update, frames=nt, interval=30, blit=True)
plt.show()