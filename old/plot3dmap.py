import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== Load file ======
filename = r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\E16_5675PLxy-4-+4step0.05_0.1s_(0.45V10_3_89uW_stablepower)09-29_20-51.h5"
with h5py.File(filename, "r") as f:
    data = f["spectro_data"][:]                 # e.g. (6561, 1, 1340)
    wvl  = f["spectro_wavelength"][:]           # (1340,)

    # Try to get explicit grid arrays if present; otherwise build from limits/steps
    if "xPositions" in f and "yPositions" in f:
        x_axis = f["xPositions"][:]
        y_axis = f["yPositions"][:]
    else:
        xlims = f["xLims"][:]
        ylims = f["yLims"][:]
        xstep = float(f["xStep"][()])
        ystep = float(f["yStep"][()])
        nx = int(round((xlims[1] - xlims[0]) / xstep)) + 1
        ny = int(round((ylims[1] - ylims[0]) / ystep)) + 1
        x_axis = np.linspace(xlims[0], xlims[1], nx)
        y_axis = np.linspace(ylims[0], ylims[1], ny)

# ====== Build the 2D map by integrating over wavelength ======
# Assume wavelength is the last axis in spectro_data
intensity_1d = np.sum(data, axis=-1).flatten()
ny, nx = len(y_axis), len(x_axis)
assert intensity_1d.size == nx * ny, f"Point count mismatch: {intensity_1d.size} vs {nx*ny}"
map2d = intensity_1d.reshape(ny, nx)

# ====== Helpers ======
def flat_index(ix, iy, nx): return iy * nx + ix
def clamp(v, lo, hi): return max(lo, min(hi, v))

# start at brightest pixel (or set to center if you prefer)
iy, ix = np.unravel_index(np.argmax(map2d), map2d.shape)
# iy, ix = ny // 2, nx // 2   # alternative: center

# ====== Figure setup ======
fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(10, 4), dpi=110)

im = ax_map.imshow(
    map2d,
    origin="lower",
    extent=[x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()],
    aspect="auto"
)
cbar = fig.colorbar(im, ax=ax_map, label="Integrated Intensity (a.u.)")
marker = ax_map.plot([x_axis[ix]], [y_axis[iy]], marker="o", ms=6)[0]
ax_map.set_xlabel("X Voltage (V)")
ax_map.set_ylabel("Y Voltage (V)")
ax_map.set_title("Use arrows / WASD (Shift = faster)")

(line,) = ax_spec.plot([], [], lw=1.4)
ax_spec.set_xlabel("Wavelength")
ax_spec.set_ylabel("Counts")

def update_plots():
    idx = flat_index(ix, iy, nx)
    spec = data[idx, 0, :]
    line.set_data(wvl, spec)
    ax_spec.relim()
    ax_spec.autoscale_view()
    ax_spec.set_title(f"Spectrum @ X={x_axis[ix]:.3f} V, Y={y_axis[iy]:.3f} V")

    marker.set_data([x_axis[ix]], [y_axis[iy]])
    fig.canvas.draw_idle()

update_plots()

# ====== Keyboard controls ======
# Arrows or WASD move one pixel; hold Shift to move 5 pixels.
def on_key(event):
    global ix, iy
    step = 5 if (event.key and "shift" in event.key) else 1

    key = event.key.replace("shift+", "") if event.key else ""
    if key in ("left", "a"):   ix = clamp(ix - step, 0, nx - 1)
    elif key in ("right", "d"):ix = clamp(ix + step, 0, nx - 1)
    elif key in ("up", "w"):   iy = clamp(iy + step, 0, ny - 1)  # y increases upward in extent
    elif key in ("down", "s"): iy = clamp(iy - step, 0, ny - 1)
    elif key == "home":        ix, iy = nx // 2, ny // 2
    elif key == "end":
        # jump to brightest pixel
        Imax = np.argmax(map2d)
        iy0, ix0 = np.unravel_index(Imax, map2d.shape)
        ix, iy = ix0, iy0
    elif key == "S":  # capital S saves spectrum (depends on backend/platform)
        # save current spectrum to CSV
        idx = flat_index(ix, iy, nx)
        spec = data[idx, 0, :]
        out = Path("spectrum_X{:.3f}_Y{:.3f}.csv".format(x_axis[ix], y_axis[iy]))
        np.savetxt(out, np.column_stack([wvl, spec]), delimiter=",", header="wavelength,counts", comments="")
        print(f"Saved: {out.resolve()}")
        return
    else:
        return

    update_plots()

fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout()
plt.show()
