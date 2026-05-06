import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. Parameters
# -------------------------
length = 10.0
c = 1.0                 # wave speed
dx = 0.05
dt = 0.045              # must satisfy c*dt <= dx
total_time = 1000
assert(c * dt <= dx)

nx = int(length / dx) + 1
x = np.linspace(0, length, nx)

s = (c * dt / dx) ** 2
print(f"CFL parameter s = {s:.4f} (must be <= 1)")
assert(s <= 1)

# -------------------------
# 2. Initial Conditions
# -------------------------
# Initial displacement φ(x)
phi = np.exp(-100 * (x - length/2)**2)

# Initial velocity ψ(x)
psi = np.zeros_like(x)

# Allocate solution arrays
u_prev = phi.copy()      # u^0
u = np.zeros_like(x)     # u^1
u_next = np.zeros_like(x)

# Compute first time step using derived formula
# u_j^1 = s/2 (φ_{j+1} + φ_{j-1}) + (1 - s)φ_j + dt ψ_j
u[1:-1] = (
    0.5 * s * (phi[2:] + phi[:-2])
    + (1 - s) * phi[1:-1]
    + dt * psi[1:-1]
)

# Dirichlet boundaries (fixed ends)
u[0] = 0
u[-1] = 0

# -------------------------
# 3. Time Stepping Loop
# -------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_ylim(-1.2, 1.2)

steps = int(total_time / dt)

for n in range(steps):

    # Stencil update (interior points)
    u_next[1:-1] = (
        s * (u[2:] + u[:-2])
        + 2 * (1 - s) * u[1:-1]
        - u_prev[1:-1]
    )

    # Dirichlet boundary conditions
    u_next[0] = 0
    u_next[-1] = 0

    # Shift time levels
    u_prev[:] = u
    u[:] = u_next

    # Visualization update
    if n % 10 == 0:
        line.set_ydata(u)
        ax.set_title(f"Time = {n*dt:.2f}")
        plt.pause(0.01)

plt.ioff()
plt.show()