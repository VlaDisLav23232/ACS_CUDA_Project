import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Parameters
length = 2       
k = 0.466        
temp_source = 200  
temp_initial = 0

dx = 0.04        # Slightly finer grid
dt = 0.0004      # Adjusted for stability
total_time = 10

nodes = int(length / dx) + 1
u = np.full((nodes, nodes), temp_initial, dtype=float)

# 2. Create Random Heat Squares
num_sources = 6
source_size = 3  # Size of the square (in nodes)
# Randomly pick top-left corners for these squares
sources = np.random.randint(source_size, nodes - source_size, size=(num_sources, 2))

def apply_heat_sources(grid):
    for (r_idx, c_idx) in sources:
        grid[r_idx:r_idx+source_size, c_idx:c_idx+source_size] = temp_source

apply_heat_sources(u)

# 3. Stability Check
r = k * dt / dx**2
print(f"Stability factor r: {r:.4f} (Must be < 0.25)")

# 4. Visualization
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(u, cmap='magma', origin='lower', extent=[0, length, 0, length], vmin=0, vmax=200)
plt.colorbar(im, label="Temperature (C˚)")

steps = int(total_time / dt)

for t in range(steps):
    u_next = u.copy()
    
    # Vectorized FTCS Calculation (Interior)
    u_next[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
        u[2:, 1:-1] + u[:-2, 1:-1] +   # y neighbors
        u[1:-1, 2:] + u[1:-1, :-2] -   # x neighbors
        4 * u[1:-1, 1:-1]
    )
    
    # INSULATED BOUNDARIES (Neumann)
    # This allows corners and edges to heat up naturally
    u_next[0, :] = u_next[1, :]   # Bottom
    u_next[-1, :] = u_next[-2, :]  # Top
    u_next[:, 0] = u_next[:, 1]   # Left
    u_next[:, -1] = u_next[:, -2]  # Right
    
    u = u_next
    
    # Re-apply heat sources so they don't cool down
    apply_heat_sources(u)

    # Update Animation
    if t % 150 == 0:
        im.set_array(u)
        ax.set_title(f"Time: {t*dt:.3f}s - Random Heat Dissipation")
        plt.draw()
        plt.pause(0.005)

plt.ioff()
plt.show()