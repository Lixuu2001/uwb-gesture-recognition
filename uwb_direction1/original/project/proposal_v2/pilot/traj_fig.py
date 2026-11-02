"""z0 sensitivity + a figure of reconstructed trajectories for the advisor."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\19252\AppData\Local\Temp\claude\D--uwb-gestures\0021ac01-3388-4472-bdc8-d381e152d862\scratchpad")
from traj_pilot import range_track, solve_dp, smooth, signed_area, dprime, CACHE

X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
y = np.load(f"{CACHE}/y_clutterremoved.npy")
subj = np.load(f"{CACHE}/subj_clutterremoved.npy")


def traj(i, z0):
    unit = np.asarray(X[i])
    rr = np.stack([smooth(range_track(unit[v])[0], 5) for v in range(3)])
    return solve_dp(rr - rr[:, :1], z0)


# ---- z0 sensitivity on the two headline readouts -------------------------
print("z0 sweep (chirality d' G10 vs G9 ; horizontal E[v^3] acc G1 vs G2), 20/subject")
for z0 in (0.3, 0.5, 0.8, 1.2):
    area = {g: [] for g in (8, 9)}
    asym = {g: [] for g in (0, 1)}
    for g in (8, 9, 0, 1):
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:20]:
                dp = traj(i, z0)
                x, yy = dp[:, 0], dp[:, 1]
                if g in (8, 9):
                    area[g].append(signed_area(x - x.mean(), yy - yy.mean()))
                else:
                    asym[g].append(np.mean((np.gradient(x) * 20) ** 3))
    d = dprime(np.array(area[9]), np.array(area[8]))
    a, b = -np.array(asym[0]), -np.array(asym[1])
    acc = 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2
    print(f"  z0={z0:4.2f} m  chirality d'={d:6.3f}   horizontal acc={acc:6.2f}%")

# ---- figure --------------------------------------------------------------
names = {0: "G1  left->right", 2: "G3  up->down", 8: "G9  clockwise", 9: "G10 anti-clockwise"}
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
rng = np.random.default_rng(0)
for ax, g in zip(axes, names):
    idx = np.where((y == g) & (subj == 1))[0][:6]
    for k, i in enumerate(idx):
        dp = traj(i, 0.5) * 1000.0
        x, yy = dp[:, 0] - dp[:, 0].mean(), dp[:, 1] - dp[:, 1].mean()
        ax.plot(x, yy, lw=1.2, alpha=0.8)
        ax.plot(x[0], yy[0], "o", ms=4, color="k", alpha=0.5)
    ax.set_title(names[g], fontsize=11)
    ax.set_xlim(-500, 500); ax.set_ylim(-500, 500)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (mm)")
axes[0].set_ylabel("y (mm)")
fig.suptitle("Hand trajectories reconstructed from 3 UWB range tracks (subject HV1, 6 repetitions each, no training)",
             fontsize=11)
fig.tight_layout()
out = r"C:\Users\19252\AppData\Local\Temp\claude\D--uwb-gestures\0021ac01-3388-4472-bdc8-d381e152d862\scratchpad\traj_demo.png"
fig.savefig(out, dpi=130)
print("saved", out)
