import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Variabler
alpha_Al   = 97e-6    # termisk diffusivitet, aluminium [m²/s]
alpha_luft = 22e-6    # termisk diffusivitet, luft [m²/s]

Lx_plate = 0.10       # platens bredde [m]
Ly_plate = 0.20       # platens høyde [m]
lx_luft  = 0.05       # luftlag i x-retning (hver side) [m]
ly_luft  = 0.10       # luftlag i y-retning (topp/bunn) [m]

Nx     = 50           # oppløsning x
Ny     = 2 * Nx       # oppløsning y
t_end  = 60           # sluttid [s]
n_snap = 8            # antall snapshots (oppgave 7a)
fps    = 20           # bilder per sekund i animasjon (oppgave 7b)


# Dimensjoner
Lx_tot = Lx_plate + 2 * lx_luft   # 0.20 m
Ly_tot = Ly_plate + 2 * ly_luft   # 0.40 m

dx = Lx_tot / (Nx + 1)
dy = Ly_tot / (Ny + 1)
x  = np.linspace(0, Lx_tot, Nx + 2)
y  = np.linspace(0, Ly_tot, Ny + 2)

# Definering av plate/luft
alpha_grid = np.ones((Nx + 2, Ny + 2)) * alpha_luft

ix0 = round(lx_luft / dx)
ix1 = round((lx_luft + Lx_plate) / dx)
iy0 = round(ly_luft / dy)
iy1 = round((ly_luft + Ly_plate) / dy)

alpha_grid[ix0:ix1+1, iy0:iy1+1] = alpha_Al

# Midtpunkt
mid_i = (ix0 + ix1) // 2
mid_j = (iy0 + iy1) // 2

# Stabilisering
dt = 0.4 / (2 * alpha_Al * (1/dx**2 + 1/dy**2))
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps
print(f"Tidssteg dt = {dt:.5f} s  |  Antall steg: {n_steps}")

# Initialbetingelser
u = np.ones((Nx + 2, Ny + 2)) * 200.0        # luft: 200°C
u[ix0:ix1+1, iy0:iy1+1] = 15.0               # plate: 15°C

# Randbetingelser
u[0, :]  = 200
u[-1, :] = 200
u[:, 0]  = 200
u[:, -1] = 200

# Simulering
snap_times = np.linspace(0, t_end, n_snap + 1)
snapshots  = []
snap_idx   = 0
time60     = None

frames_for_anim = []
frame_interval  = max(1, n_steps // (fps * 10))

rx = dt / dx**2
ry = dt / dy**2

for step in range(n_steps + 1):
    t = step * dt

    if snap_idx <= n_snap and t >= snap_times[snap_idx] - 1e-9:
        snapshots.append((t, u.copy()))
        snap_idx += 1

    if time60 is None and u[mid_i, mid_j] >= 60:
        time60 = t

    if step % frame_interval == 0:
        frames_for_anim.append((t, u.copy()))

    if step == n_steps:
        break

    u_new = u.copy()
    u_new[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + alpha_grid[1:-1, 1:-1] * rx
          * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        + alpha_grid[1:-1, 1:-1] * ry
          * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
    )
    u_new[0, :]  = 200
    u_new[-1, :] = 200
    u_new[:, 0]  = 200
    u_new[:, -1] = 200
    u = u_new

if time60 is not None:
    print(f"\nOppgave 7c: Midten av platen når 60°C etter {time60:.1f} s ({time60/60:.1f} min)")
else:
    print(f"\nOppgave 7c: Midten nådde ikke 60°C innen {t_end} s – øk t_end")

def draw_plate_outline(ax):
    "Tegner en hvit stiplet ramme rundt platen i plottet."
    rect = Rectangle(
        (x[ix0]*100, y[iy0]*100),
        Lx_plate * 100, Ly_plate * 100,
        linewidth=1.2, edgecolor="white", facecolor="none",
        linestyle="--", alpha=0.6
    )
    ax.add_patch(rect)

# Bilder
n_plots = len(snapshots)
cols = min(4, n_plots)
rows = int(np.ceil(n_plots / cols))

fig = plt.figure(figsize=(cols * 3.6 + 0.8, rows * 5.0 + 0.8))
fig.patch.set_facecolor("#0a0e1a")
gs = GridSpec(rows, cols + 1, figure=fig,
              width_ratios=[1] * cols + [0.06],
              hspace=0.50, wspace=0.30,
              left=0.07, right=0.92, top=0.91, bottom=0.06)

cbar_ax = fig.add_subplot(gs[:, -1])
axes = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]

im = None
for i, (t, grid) in enumerate(snapshots):
    ax = axes[i]
    ax.set_facecolor("#111827")
    mid_temp = grid[mid_i, mid_j]
    im = ax.imshow(grid.T, origin="lower", vmin=15, vmax=200,
                   cmap="RdBu_r",
                   extent=[0, Lx_tot*100, 0, Ly_tot*100],
                   aspect="auto")

    draw_plate_outline(ax)
    ax.plot(x[mid_i]*100, y[mid_j]*100, "w+", markersize=10, markeredgewidth=2)

    is_60 = (time60 is not None and abs(t - time60) < dt * 2)
    ax.set_title(f"t = {t:.1f} s  |  plate: {mid_temp:.1f}°C",
                 color="limegreen" if is_60 else "white", fontsize=8.5, pad=5)

    r_idx, c_idx = divmod(i, cols)
    ax.set_xlabel("Bredde x [cm]" if r_idx == rows - 1 or i >= n_plots - cols else "",
                  fontsize=8, color="#94a3b8")
    ax.set_ylabel("Høyde y [cm]" if c_idx == 0 else "", fontsize=8, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor("limegreen" if is_60 else "#1e3a5f")
        spine.set_linewidth(2 if is_60 else 0.8)

for j in range(n_plots, len(axes)):
    axes[j].set_visible(False)

cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Temperatur [°C]", color="#94a3b8", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#64748b", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")

plt.suptitle(
    f"Oppgave 7a – Plate (10×20 cm) med luftlag  |  stiplet = platekant",
    color="#7dd3fc", fontsize=12, y=0.97
)

# Animasjon
fig2, ax2 = plt.subplots(figsize=(4.5, 8))
fig2.patch.set_facecolor("#0a0e1a")
ax2.set_facecolor("#111827")

im2 = ax2.imshow(frames_for_anim[0][1].T, origin="lower", vmin=15, vmax=200,
                 cmap="RdBu_r",
                 extent=[0, Lx_tot*100, 0, Ly_tot*100],
                 aspect="auto", animated=True)

draw_plate_outline(ax2)
ax2.plot(x[mid_i]*100, y[mid_j]*100, "w+", markersize=12, markeredgewidth=2)

title2 = ax2.set_title("t = 0.0 s", color="#7dd3fc", fontsize=11)
ax2.set_xlabel("Bredde x [cm]", color="#94a3b8")
ax2.set_ylabel("Høyde y [cm]", color="#94a3b8")
ax2.tick_params(colors="#64748b")
plt.colorbar(im2, ax=ax2, label="Temperatur [°C]")
plt.tight_layout()

def update(i):
    t, grid = frames_for_anim[i]
    mid_temp = grid[mid_i, mid_j]
    im2.set_data(grid.T)
    color = "limegreen" if mid_temp >= 60 else "#7dd3fc"
    title2.set_text(f"t = {t:.1f} s  |  plate: {mid_temp:.1f}°C")
    title2.set_color(color)
    return im2, title2

ani = animation.FuncAnimation(fig2, update, frames=len(frames_for_anim),
                               interval=1000/fps, blit=True, repeat=True)

plt.suptitle("Oppgave 7b – Animasjon med luftlag", color="#7dd3fc", fontsize=12)
plt.show()
