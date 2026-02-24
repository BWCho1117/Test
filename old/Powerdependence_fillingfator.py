# filepath: c:\Users\bwcho\OneDrive\HWU\Projects\Dipolar ladder excitons_wTatyana\Code\Powerdependence_fillingfator.py
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
# SciPy 여부 확인
try:
    from scipy.interpolate import PchipInterpolator, CubicSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ====== 기존 데이터/함수 (필요 부분 복사) ======
data_raw = {
    0.007: { 0.0: {"top": (-0.536, None)}, -1/3: {"top": -1.030}, -2/3: {"top": -1.360}, -1.0: {"top": -1.607}, -2.0: {"top": -2.102}, },
    0.6:   { 0.0: {"top": (-0.622, -0.619)}, -1/3: {"top": -1.095}, -2/3: {"top": -1.444}, -1.0: {"top": -1.609}, },
    1.5:   { 0.0: {"top": (-0.683, None)}, -1/3: {"top": -1.156}, -2/3: {"top": -1.465}, -1.0: {"top": -1.609}, },
    20.0:  { 0.0: {"top": (-0.807, -0.868)}, -1.0: {"top": -1.609}, },
    70.0:  { 0.0: {"top": (-0.807, -1.115)}, -1.0: {"top": -1.609}, },
}

def _normalize_range(r):
    a, b = r
    if b is None: return a, a
    lo, hi = sorted([a, b])
    return lo, hi

def _interp_with_extrap(xq, xp, yp):
    if xq <= xp[0]:
        if len(xp) > 1:
            m = (yp[1]-yp[0])/(xp[1]-xp[0]); return yp[0] + m*(xq-xp[0])
        return yp[0]
    if xq >= xp[-1]:
        if len(xp) > 1:
            m = (yp[-1]-yp[-2])/(xp[-1]-xp[-2]); return yp[-1] + m*(xq-xp[-1])
        return yp[-1]
    return float(np.interp(xq, xp, yp))

def get_top_voltage(power_uW: float, filling: float, mode="mid", extrapolate=False):
    powers = np.array(sorted(data_raw.keys()))
    def handle(xp, yp):
        return _interp_with_extrap(power_uW, xp, yp) if extrapolate else float(np.interp(np.clip(power_uW, xp.min(), xp.max()), xp, yp))
    if filling == 0.0:
        mins, mids, maxs, ps = [], [], [], []
        for p in powers:
            e = data_raw[p].get(0.0)
            if e:
                vmin, vmax = _normalize_range(e["top"] if isinstance(e["top"], tuple) else (e["top"], None))
                ps.append(p); mins.append(vmin); maxs.append(vmax); mids.append(0.5*(vmin+vmax))
        if not ps: raise ValueError("No ν=0 data")
        ps = np.array(ps); mins = np.array(mins); maxs = np.array(maxs); mids = np.array(mids)
        if mode=="min": return handle(ps, mins)
        if mode=="max": return handle(ps, maxs)
        return handle(ps, mids)
    else:
        ps, vs = [], []
        for p in powers:
            e = data_raw[p].get(filling)
            if e:
                val = e["top"][0] if isinstance(e["top"], tuple) else e["top"]
                ps.append(p); vs.append(val)
        if not ps: raise ValueError(f"No data for ν={filling}")
        ps = np.array(ps); vs = np.array(vs)
        return handle(ps, vs)

def list_fillings():
    s = set()
    for pw in data_raw.values(): s.update(pw.keys())
    return sorted(s)

# ====== GUI ======
class FillingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filling Factor Gate Voltage Helper")
        self.geometry("640x420")
        self.fillings_all = list_fillings()
        # Top input frame
        frm = ttk.Frame(self, padding=8)
        frm.pack(fill="x")
        ttk.Label(frm, text="Power (µW or comma list):").grid(row=0, column=0, sticky="w")
        self.power_entry = ttk.Entry(frm, width=30)
        self.power_entry.insert(0, "5, 10, 25")
        self.power_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(frm, text="Neutral mode:").grid(row=0, column=2, padx=(20,2))
        self.neutral_mode = tk.StringVar(value="mid")
        ttk.Combobox(frm, textvariable=self.neutral_mode, values=["min","mid","max"], width=5).grid(row=0, column=3)

        self.var_extrap = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Extrapolate power", variable=self.var_extrap).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text="Gate angle (deg):").grid(row=1, column=2, sticky="e")
        self.angle_entry = ttk.Entry(frm, width=6)
        self.angle_entry.insert(0,"40")
        self.angle_entry.grid(row=1, column=3, sticky="w")

        # Filling selection
        fs = ttk.LabelFrame(self, text="Fillings (ν)")
        fs.pack(fill="x", padx=8, pady=4)
        self.fill_vars = {}
        row=0; col=0
        for f in self.fillings_all:
            v = tk.BooleanVar(value=(f in [0.0,-1/3,-2/3,-1.0]))
            cb = ttk.Checkbutton(fs, text=f"{f:g}", variable=v)
            cb.grid(row=row, column=col, sticky="w", padx=4, pady=2)
            self.fill_vars[f] = v
            col += 1
            if col>8: col=0; row+=1

        # Buttons
        btn_frm = ttk.Frame(self, padding=4)
        btn_frm.pack(fill="x")
        ttk.Button(btn_frm, text="Compute", command=self.compute).pack(side="left")
        ttk.Button(btn_frm, text="Copy Table", command=self.copy_table).pack(side="left", padx=6)
        ttk.Button(btn_frm, text="Quit", command=self.destroy).pack(side="right")
        ttk.Button(btn_frm, text="Plot Fits", command=self.plot_fits).pack(side="left", padx=6)

        # 추가: Fit 옵션 / Log scale
        opt_frm = ttk.Frame(self, padding=(8,2))
        opt_frm.pack(fill="x")
        ttk.Label(opt_frm, text="Fit method:").grid(row=0, column=0, sticky="w")
        self.fit_method = tk.StringVar(value="linear")
        fit_vals = ["linear"]
        if _HAVE_SCIPY:
            fit_vals += ["pchip","cspline"]
        ttk.Combobox(opt_frm, textvariable=self.fit_method, values=fit_vals, width=8).grid(row=0, column=1, sticky="w", padx=(2,10))
        self.var_logx = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frm, text="Log Power axis", variable=self.var_logx).grid(row=0, column=2, sticky="w")
        ttk.Label(opt_frm, text="Dense points:").grid(row=0, column=3, padx=(15,2))
        self.dense_entry = ttk.Entry(opt_frm, width=6)
        self.dense_entry.insert(0,"300")
        self.dense_entry.grid(row=0, column=4, sticky="w")

        # Matplotlib Figure 영역
        self.fig = plt.Figure(figsize=(6.2,4.0), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        # === Results table & status ===
        cols = ("power","filling","topV","bottomV")
        self.result_tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        for c,w in zip(cols,(90,80,110,110)):
            self.result_tree.heading(c, text=c)
            self.result_tree.column(c, width=w, anchor="center")
        self.result_tree.pack(fill="x", padx=6, pady=(2,4))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=6, pady=(0,4))

    def parse_powers(self):
        txt = self.power_entry.get().strip()
        if not txt: raise ValueError("Enter power values.")
        arr=[]
        for part in txt.split(","):
            try:
                arr.append(float(part.strip()))
            except:
                raise ValueError(f"Bad power token: {part}")
        return arr

    def selected_fillings(self):
        return [f for f,v in self.fill_vars.items() if v.get()]

    def _calc_rows(self, powers, fillings, neutral_mode, extrapolate, angle_deg):
        """공통 계산: (power, filling, top, bottom) 리스트 반환"""
        tan_angle = math.tan(math.radians(angle_deg))
        rows = []
        for p in powers:
            for f in fillings:
                topV = get_top_voltage(p, f, mode=neutral_mode, extrapolate=extrapolate)
                bottomV = topV / tan_angle
                rows.append((p, f, topV, bottomV))
        return rows

    def _update_table(self, rows):
        self.result_tree.delete(*self.result_tree.get_children())
        for r in rows:
            self.result_tree.insert("", "end",
                                    values=(f"{r[0]:.6g}", f"{r[1]:g}",
                                            f"{r[2]:.5f}", f"{r[3]:.5f}"))

    def compute(self):
        try:
            powers = self.parse_powers()
            fillings = self.selected_fillings()
            if not fillings:
                raise ValueError("Select at least one filling.")
            angle_deg = float(self.angle_entry.get())
            nm = self.neutral_mode.get()
            extrap = self.var_extrap.get()
            rows = self._calc_rows(powers, fillings, nm, extrap, angle_deg)
            self._update_table(rows)
            self.status.set(f"Computed {len(rows)} rows.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set(f"Error: {e}")

    def copy_table(self):
        rows = [self.result_tree.item(i,"values") for i in self.result_tree.get_children()]
        if not rows:
            self.status.set("Nothing to copy.")
            return
        header = "Power_uW\tFilling\tTop_V\tBottom_V"
        lines = [header] + ["\t".join(r) for r in rows]
        txt = "\n"+ "\n".join(lines) + "\n"
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.status.set("Table copied to clipboard.")

    # === 새: 곡선 피팅용 내부 함수 ===
    def _fit_curve(self, x, y, x_dense, method):
        x = np.asarray(x); y = np.asarray(y)
        # 정렬
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]
        if len(x) == 1:
            return np.full_like(x_dense, y[0], dtype=float)
        if method == "linear" or not _HAVE_SCIPY:
            return np.interp(x_dense, x, y)
        if method == "pchip":
            if len(x) < 2:
                return np.interp(x_dense, x, y)
            try:
                f = PchipInterpolator(x, y)
                return f(x_dense)
            except Exception:
                return np.interp(x_dense, x, y)
        if method == "cspline":
            if len(x) < 3:   # CubicSpline 최소 2~3 필요
                return np.interp(x_dense, x, y)
            try:
                f = CubicSpline(x, y)
                return f(x_dense)
            except Exception:
                return np.interp(x_dense, x, y)
        return np.interp(x_dense, x, y)

    def plot_fits(self):
        try:
            fillings = self.selected_fillings()
            if not fillings:
                raise ValueError("Select fillings first.")
            powers_all = sorted(data_raw.keys())
            fit_method = self.fit_method.get()
            dense_n = max(50, int(self.dense_entry.get()))
            p_min = min(powers_all); p_max = max(powers_all)
            if self.var_logx.get():
                x_dense = np.logspace(np.log10(p_min), np.log10(p_max), dense_n)
            else:
                x_dense = np.linspace(p_min, p_max, dense_n)

            self.ax.clear()

            # --- ν=0 band ---
            if 0.0 in fillings:
                p_list = []; vmin_list = []; vmax_list = []; vmid_list = []
                for p in powers_all:
                    e = data_raw[p].get(0.0)
                    if not e: continue
                    if isinstance(e["top"], tuple):
                        vmin, vmax = _normalize_range(e["top"])
                    else:
                        vmin = vmax = e["top"]
                    vmid = 0.5*(vmin+vmax)
                    p_list.append(p); vmin_list.append(vmin); vmax_list.append(vmax); vmid_list.append(vmid)
                if p_list:
                    vmin_fit = self._fit_curve(p_list, vmin_list, x_dense, fit_method)
                    vmax_fit = self._fit_curve(p_list, vmax_list, x_dense, fit_method)
                    vmid_fit = self._fit_curve(p_list, vmid_list, x_dense, fit_method)
                    self.ax.fill_between(x_dense, vmin_fit, vmax_fit,
                                         color="gold", alpha=0.25, label="ν=0 range fit")
                    self.ax.plot(x_dense, vmid_fit, color="goldenrod", lw=2,
                                 label="ν=0 mid fit")
                    self.ax.plot(p_list, vmid_list, 'o', color="goldenrod", ms=5, alpha=0.85)

            # --- Other fillings ---
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                                ['tab:blue','tab:orange','tab:green','tab:red','tab:purple'])
            for k,f in enumerate(fillings):
                if f == 0.0: continue
                p_pts=[]; v_pts=[]
                for p in powers_all:
                    e = data_raw[p].get(f)
                    if e:
                        val = e["top"][0] if isinstance(e["top"], tuple) else e["top"]
                        p_pts.append(p); v_pts.append(val)
                if not p_pts: continue
                c = color_cycle[k % len(color_cycle)]
                v_fit = self._fit_curve(p_pts, v_pts, x_dense, fit_method)
                self.ax.plot(x_dense, v_fit, '-', color=c, lw=2, label=f"ν={f:g} fit")
                self.ax.plot(p_pts, v_pts, 'o', color=c, ms=5)

            # --- Query points (powers entered) + table update ---
            angle_deg = float(self.angle_entry.get())
            nm = self.neutral_mode.get()
            extrap = self.var_extrap.get()
            try:
                q_powers = self.parse_powers()
            except:
                q_powers = []
            if q_powers:
                rows = self._calc_rows(q_powers, fillings, nm, extrap, angle_deg)
                self._update_table(rows)  # 표 갱신
                # Plot verticals and markers
                for qp in q_powers:
                    self.ax.axvline(qp, color="k", ls="--", lw=0.8, alpha=0.4)
                # Marker per (power,filling)
                for (p,f,topV,_) in rows:
                    self.ax.plot(p, topV, marker='s', ms=6,
                                 color='black', mec='white', mew=0.7, zorder=6)
                # 간단한 텍스트 (첫 power / 첫 filling 만 요약)
                txt_lines = [f"P={q_powers[0]:.3g}µW"]
                for (p,f,topV,_) in rows:
                    if abs(p - q_powers[0]) < 1e-9:
                        txt_lines.append(f"ν={f:g}: {topV:.3f}V")
                self.ax.text(0.02, 0.02, "\n".join(txt_lines),
                             transform=self.ax.transAxes,
                             fontsize=9, color="black",
                             bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=4))

            self.ax.set_xlabel("Optical Power (µW)")
            self.ax.set_ylabel("Top Voltage (V)")
            self.ax.set_title(f"Top Voltage vs Power (fit={fit_method})")
            if self.var_logx.get():
                self.ax.set_xscale("log")
            self.ax.grid(alpha=0.3)
            self.ax.legend(frameon=False, fontsize=9, ncol=2)
            self.fig.tight_layout()
            self.canvas.draw()
            if q_powers:
                self.status.set(f"Plot updated. Rows: {len(rows)}")
            else:
                self.status.set("Plot updated.")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))
            self.status.set(f"Plot error: {e}")

if __name__ == "__main__":
    app = FillingGUI()
    app.mainloop()