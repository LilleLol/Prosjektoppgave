import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

alpha  = 97e-6   # termisk diffusivitet for aluminium [m²/s]
Lx     = 0.10    # bredde [m]
Ly     = 0.20    # høyde [m]
Nx     = 25      # oppløsning x
Ny     = 2 * Nx  # oppløsning y
t_end  = 60      # sluttid [s]
n_snap = 8       # antall bilder
fps    = 20      # bilder per sekund i animasjonen

dx = Lx / (Nx + 1)
dy = Ly / (Ny + 1)
x  = np.linspace(0, Lx, Nx + 2)
y  = np.linspace(0, Ly, Ny + 2)

# Stabilitetskrav
dt = 0.4 / (2 * alpha * (1/dx**2 + 1/dy**2))
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

print(f"Tidssteg dt = {dt:.4f} s  |  Antall steg: {n_steps}")

# Initialtemperatur og randbetingelser
u = np.ones((Nx + 2, Ny + 2)) * 15.0
u[0, :]  = 200
u[-1, :] = 200
u[:, 0]  = 200
u[:, -1] = 200

mid_i = (Nx + 2) // 2
mid_j = (Ny + 2) // 2

# Bilder
snap_times = np.linspace(0, t_end, n_snap + 1)
snapshots  = []
snap_idx   = 0
time60     = None

# Animasjon
frames_for_anim = []
frame_interval  = max(1, n_steps // (fps * 10))

rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

for step in range(n_steps + 1):
    t = step * dt

    # Lagre bilder
    if snap_idx <= n_snap and t >= snap_times[snap_idx] - 1e-9:
        snapshots.append((t, u.copy()))
        snap_idx += 1

    # Sjekk om midten har nådd 60°C
    if time60 is None and u[mid_i, mid_j] >= 60:
        time60 = t
        snap_time60 = u.copy()

    # Lagre bilde til animasjon
    if step % frame_interval == 0:
        frames_for_anim.append((t, u.copy()))

    if step == n_steps:
        break

    # Forlengs Euler
    u_new = u.copy()
    u_new[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + rx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        + ry * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
    )
    # Randbetingelser
    u_new[0, :] = u_new[-1, :] = 200
    u_new[:, 0] = u_new[:, -1] = 200
    u = u_new

# Bilder
n_plots = len(snapshots)
cols = min(4, n_plots)
rows = int(np.ceil(n_plots / cols))

# GridSpec
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(cols * 3.4 + 0.8, rows * 4.2 + 0.8))
fig.patch.set_facecolor("#0a0e1a")
gs = GridSpec(rows, cols + 1, figure=fig,
              width_ratios=[1] * cols + [0.06],
              hspace=0.45, wspace=0.32,
              left=0.07, right=0.92, top=0.91, bottom=0.06)

cbar_ax = fig.add_subplot(gs[:, -1])
axes = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]

im = None
for i, (t, grid) in enumerate(snapshots):
    ax = axes[i]
    ax.set_facecolor("#111827")
    mid_temp = grid[mid_i, mid_j]
    im = ax.imshow(grid.T, origin="lower", vmin=15, vmax=200,
                   cmap="RdBu_r", extent=[0, Lx*100, 0, Ly*100], aspect="auto")

    # Midtpunkt-markør
    ax.plot(x[mid_i]*100, y[mid_j]*100, "w+", markersize=10, markeredgewidth=2)

    is_60 = (time60 is not None and abs(t - time60) < dt * 2)
    title_color = "limegreen" if is_60 else "white"
    ax.set_title(f"t = {t:.1f} s  |  midten: {mid_temp:.1f}°C",
                 color=title_color, fontsize=8.5, pad=5)

    # Akseetiketter bare på ytterkanter
    r, c = divmod(i, cols)
    ax.set_xlabel("Bredde x [cm]" if r == rows - 1 or i >= n_plots - cols else "",
                  fontsize=8, color="#94a3b8")
    ax.set_ylabel("Høyde y [cm]" if c == 0 else "", fontsize=8, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor("limegreen" if is_60 else "#1e3a5f")
        spine.set_linewidth(2 if is_60 else 0.8)

# Skjul tomme subplots
for j in range(n_plots, len(axes)):
    axes[j].set_visible(False)

# Én felles colorbar i den dedikerte kolonnen
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Temperatur [°C]", color="#94a3b8", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#64748b", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")

plt.suptitle("Oppgave 6c – Varmefordeling i aluminiumsplate (10 × 20 cm)",
             color="#7dd3fc", fontsize=13, y=0.97)

# Når temperatur er 60°C
if time60 is not None:
    print(f"\nOppgave 6c: Midten når 60°C etter {time60:.1f} s ({time60/60:.1f} min)")
else:
    print(f"\nOppgave 6c: Midten nådde ikke 60°C innen {t_end} s – øk t_end")

# Animasjon plot
fig2, ax2 = plt.subplots(figsize=(4, 7))
fig2.patch.set_facecolor("#0a0e1a")
ax2.set_facecolor("#111827")

im2 = ax2.imshow(frames_for_anim[0][1].T, origin="lower", vmin=15, vmax=200,
                 cmap="RdBu_r", extent=[0, Lx*100, 0, Ly*100], aspect="auto", animated=True)
marker, = ax2.plot(x[mid_i]*100, y[mid_j]*100, "w+", markersize=12, markeredgewidth=2)
title = ax2.set_title("t = 0.0 s", color="#7dd3fc", fontsize=11)
ax2.set_xlabel("Bredde x [cm]", color="#94a3b8")
ax2.set_ylabel("Høyde y [cm]", color="#94a3b8")
ax2.tick_params(colors="gray")
plt.colorbar(im2, ax=ax2, label="Temperatur [°C]")
plt.tight_layout()

def update(i):
    t, grid = frames_for_anim[i]
    mid_temp = grid[mid_i, mid_j]
    im2.set_data(grid.T)
    color = "limegreen" if mid_temp >= 60 else "#7dd3fc"
    title.set_text(f"t = {t:.1f} s  |  midten: {mid_temp:.1f}°C")
    title.set_color(color)
    return im2, title

ani = animation.FuncAnimation(fig2, update, frames=len(frames_for_anim),
                               interval=1000/fps, blit=True, repeat=True)

plt.suptitle("Oppgave 6d – Animasjon", color="#7dd3fc", fontsize=12)
plt.show()
