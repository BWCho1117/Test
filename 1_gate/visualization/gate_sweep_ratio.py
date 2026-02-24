import h5py
import numpy as np
import matplotlib
import os
# NEW: Windows에서 인터랙티브 백엔드 보장 (pyplot import 전에)
if os.name == "nt":
    try:
        be = matplotlib.get_backend().lower()
    except Exception:
        be = ""
    if ("inline" in be) or be.endswith("agg"):
        matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy import constants
from tkinter import Tk, filedialog
# 추가
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.widgets import Button
import io
from scipy.ndimage import gaussian_filter1d
# NEW: copy_png에서 사용
import datetime, struct, time

# 이미지 클립보드(Windows) 지원
try:
    from PIL import Image
    import win32clipboard, win32con
    _WIN_CLIP = True
except Exception:
    _WIN_CLIP = False

# DEBUG: 백엔드/클립보드 상태 로깅
print(f"[FigureCopy] backend={matplotlib.get_backend()}, win_clip={_WIN_CLIP}")

def _rgba_to_dib_bi_rgb(rgba: np.ndarray) -> bytes:
    """
    RGBA(HxWx4) → CF_DIB용 BITMAPINFOHEADER(40 bytes, BI_RGB, 32bpp) + BGRA top-down 바이트열
    - 가장 호환성이 좋음(Word, PPT, Paint 등)
    """
    h, w, _ = rgba.shape
    bgra = rgba[..., [2, 1, 0, 3]].copy()  # RGBA → BGRA
    header_size = 40  # BITMAPINFOHEADER
    image_size = w * h * 4
    # top-down DIB: height 음수
    header = struct.pack(
        "<IiiHHIIiiII",
        header_size,     # biSize
        w, -h,           # biWidth, biHeight(top-down)
        1, 32,           # biPlanes, biBitCount
        0,               # biCompression = BI_RGB
        image_size,      # biSizeImage
        2835, 2835,      # biXPelsPerMeter, biYPelsPerMeter (≈72 DPI)
        0, 0             # biClrUsed, biClrImportant
    )
    return header + bgra.tobytes()

def copy_png_to_clipboard(fig, dpi=150):
    """
    Copy figure to Windows clipboard as CF_DIB (BITMAPINFOHEADER/BI_RGB).
    Fallback: save PNG on Desktop.
    """
    def _fig_rgba(_fig) -> np.ndarray:
        _fig.canvas.draw()
        return np.asarray(_fig.canvas.buffer_rgba(), dtype=np.uint8)

    try:
        rgba = _fig_rgba(fig)
        dib = _rgba_to_dib_bi_rgb(rgba)

        if not _WIN_CLIP:
            raise RuntimeError("pywin32 not available")

        # 여러 앱이 클립보드를 잠글 수 있어 짧게 재시도
        for attempt in range(8):
            try:
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(win32con.CF_DIB, dib)
                finally:
                    win32clipboard.CloseClipboard()
                print("[FigureCopy] Copied to clipboard (CF_DIB, BI_RGB).")
                return True
            except Exception as ce:
                if attempt == 7:
                    raise
                time.sleep(0.15)
    except Exception as e:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.expanduser("~"), "Desktop", f"figure_{ts}.png")
        try:
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"[FigureCopy] Clipboard copy failed: {e}\nSaved to: {out}")
        except Exception as e2:
            print(f"[FigureCopy] Copy & save failed: {e} / {e2}")
        return False

# 폰트 크기 기본값(원하는 값으로 조정)
AX_LABEL_FONTSIZE = 13
AX_TICK_FONTSIZE  = 11
CB_LABEL_FONTSIZE = 10
CB_TICK_FONTSIZE  = 9

# 텍스트 배지 기반 Copy PNG (widgets.Button 대신 항상 보이는 방법)
def attach_copy_button(fig, ax=None, label="Copy PNG (c)"):
    """
    우상단 텍스트 배지를 추가하고 클릭/키(c)로 PNG 복사.
    - ax가 주어지면 축 좌표계(0~1) 기준 우상단에 배지 표시 → 항상 보임.
    """
    if ax is not None:
        trans = ax.transAxes
        x, y = 0.98, 0.98
    else:
        trans = fig.transFigure
        x, y = 0.98, 0.98

    txt = fig.text(
        x, y, label, ha="right", va="top",
        fontsize=10,
        bbox=dict(fc="white", ec="0.4", alpha=0.85, boxstyle="round,pad=0.25"),
        zorder=2000, transform=trans
    )

    def _do_copy():
        ok = copy_png_to_clipboard(fig)
        try:
            txt.set_text("Copied!" if ok else "Saved to Desktop")
            fig.canvas.draw_idle()
            t = fig.canvas.new_timer(interval=1200)
            t.add_callback(lambda: (txt.set_text(label), fig.canvas.draw_idle()))
            t.start()
        except Exception:
            pass

    def _on_press(event):
        try:
            renderer = fig.canvas.get_renderer()
            bb = txt.get_window_extent(renderer=renderer)
            if bb.contains(event.x, event.y):
                _do_copy()
        except Exception:
            pass

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("key_press_event", lambda e: _do_copy() if getattr(e, "key", "").lower() == "c" else None)
    return txt

# --- helper: add top axis in wavelength (nm) mapped from energy (eV) ---
def add_top_wavelength_axis(ax, label="Wavelength (nm)"):
    """
    상단 보조 x축에 파장(nm)을 표시. Energy(eV) <-> Wavelength(nm) 양방향 변환을 연결.
    λ[nm] = 1240 / E[eV]
    """
    def e2nm(E):
        E = np.asarray(E, dtype=float)
        return 1240.0 / np.maximum(E, 1e-12)          # <- safe divide
    def nm2e(nm):
        nm = np.asarray(nm, dtype=float)
        return 1240.0 / np.maximum(nm, 1e-12)         # <- safe divide
    secax = ax.secondary_xaxis('top', functions=(e2nm, nm2e))
    secax.set_xlabel(label, fontsize=AX_LABEL_FONTSIZE)
    secax.tick_params(labelsize=AX_TICK_FONTSIZE-1)
    return secax

# === MOVE HERE: helpers used by run_analysis ===
def edges_from_centers(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        step = 1.0
        return np.array([x[0] - step/2, x[0] + step/2], dtype=float)
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + dx/2.0
    edges[0] = x[0] - dx[0]/2.0
    edges[-1] = x[-1] + dx[-1]/2.0
    return edges

def attach_spec_y_lock(fig, ax, state, pos=(0.82, 0.98)):
    """스펙트럼 창에 Y‑lock 배지를 달고 클릭으로 토글."""
    def _label():
        return "Y-lock: On" if state.get("spec_y_lock", True) else "Y-lock: Off"
    txt = fig.text(
        pos[0], pos[1], _label(), transform=fig.transFigure,
        ha="left", va="top", fontsize=10,
        bbox=dict(fc="white", ec="0.4", alpha=0.85, boxstyle="round,pad=0.25"),
        zorder=2100
    )
    def _toggle():
        lock = not state.get("spec_y_lock", True)
        state["spec_y_lock"] = lock
        if lock:
            state["spec_ylim"] = ax.get_ylim()
        txt.set_text(_label()); fig.canvas.draw_idle()
    def _on_press(event):
        try:
            bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            if bb.contains(event.x, event.y):
                _toggle()
        except Exception:
            pass
    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("key_press_event",
        lambda e: _toggle() if getattr(e, "key", "").lower() == "l" and e.canvas.figure is fig else None)
    return txt
# === end helpers ===

# --- DEVICE & PHYSICAL CONSTANTS ---
epsilon_r = 3.0
V_cnp = -0.4
dt = 9e-9
db = 12e-9
e = constants.elementary_charge
epsilon_0 = constants.epsilon_0

def voltage_to_density(vg):
    delta_vg = vg - V_cnp
    n_m2 = (epsilon_r * epsilon_0 * delta_vg / e) * (1/dt + 1/db)
    n_cm2 = n_m2 / 1e4
    return n_cm2 / 1e12

# NEW: density(10^12 cm^-2) -> gate voltage(V)
def density_to_voltage(n12):
    try:
        n_m2 = float(n12) * 1e12 * 1e4  # 10^12 cm^-2 -> m^-2
        delta_vg = (n_m2 * e) / (epsilon_r * epsilon_0) / (1/dt + 1/db)
        return float(delta_vg + V_cnp)
    except Exception:
        return None

# NEW: "1/3, 0.5, 2" 같이 입력된 문자열을 float 리스트로 파싱
def parse_number_list(s: str):
    vals = []
    if not s:
        return vals
    for tok in s.replace(';', ',').split(','):
        t = tok.strip()
        if not t: 
            continue
        try:
            if '/' in t:
                num, den = t.split('/', 1)
                vals.append(float(num) / float(den))
            else:
                vals.append(float(t))
        except Exception:
            pass
    return vals

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# NEW: figure 안에서 강도 조절 단축키
def attach_intensity_controls(fig, ax, im, data=None):
    """
    [ / ]: vmin -, +
    - / =: vmax -, +
    p: percentile auto, r: reset
    MidpointNormalize는 0 중심의 ±범위만 조절.
    """
    is_log = isinstance(im.norm, mcolors.LogNorm)
    is_mid = isinstance(im.norm, MidpointNormalize)

    if hasattr(im, "norm") and im.norm is not None:
        orig_vmin, orig_vmax = float(im.norm.vmin), float(im.norm.vmax)
    else:
        orig_vmin, orig_vmax = im.get_clim()

    osd = ax.text(0.02, 0.02, "", transform=ax.transAxes, ha="left", va="bottom",
                  fontsize=9, color="k",
                  bbox=dict(fc="white", ec="0.4", alpha=0.75, boxstyle="round,pad=0.25"),
                  zorder=1500)

    def _get():
        if hasattr(im, "norm") and im.norm is not None:
            return float(im.norm.vmin), float(im.norm.vmax)
        vmin, vmax = im.get_clim(); return float(vmin), float(vmax)

    def _show():
        vmin, vmax = _get()
        osd.set_text(
            (f"±{max(abs(vmin),abs(vmax)):.3g}  (-/= range, r reset, p auto)" if is_mid
             else f"{'log ' if is_log else ''}[{vmin:.3g}, {vmax:.3g}]  ([ ] min, -/= max, r reset, p auto)")
        )

    def _apply(vmin, vmax):
        if is_log:
            vmin = max(1e-12, float(vmin)); vmax = max(vmin*1.0001, float(vmax))
            if hasattr(im, "norm") and im.norm is not None:
                im.norm.vmin = vmin; im.norm.vmax = vmax; im.changed()
            else:
                im.set_norm(mcolors.LogNorm(vmin=vmin, vmax=vmax))
        elif is_mid:
            V = max(abs(vmin), abs(vmax)); im.norm.vmin = -V; im.norm.vmax = +V; im.changed()
        else:
            im.set_clim(vmin, vmax)
        _show(); fig.canvas.draw_idle()

    def _auto():
        if data is None: return
        arr = np.asarray(data); arr = arr[np.isfinite(arr)]
        if arr.size == 0: return
        if is_log:
            arr = arr[arr > 0]; 
            if arr.size == 0: return
            vmin = float(np.percentile(arr, 50)); vmax = float(np.percentile(arr, 99.5))
        elif is_mid:
            V = float(np.percentile(np.abs(arr), 99.0)); vmin, vmax = -V, +V
        else:
            vmin = float(np.percentile(arr, 1.0)); vmax = float(np.percentile(arr, 99.5))
        _apply(vmin, vmax)

    def _reset(): _apply(orig_vmin, orig_vmax)

    def _on_key(e):
        if e.canvas.figure is not fig: return
        vmin, vmax = _get(); step = 0.10; span = max(1e-15, (vmax - vmin))
        if e.key == 'r': _reset(); return
        if e.key == 'p': _auto(); return
        if is_mid:
            V = max(abs(vmin), abs(vmax))
            if e.key in ('-', '_'): _apply(-max(1e-15, V*(1-step)), +max(1e-15, V*(1-step)))
            elif e.key in ('=', '+'): _apply(-(V*(1+step)), +(V*(1+step)))
            return
        if e.key == '[':  _apply(vmin - (span*step if not isinstance(im.norm, mcolors.LogNorm) else vmin*step), vmax)
        elif e.key == ']': _apply(vmin + (span*step if not isinstance(im.norm, mcolors.LogNorm) else vmin*step), vmax)
        elif e.key in ('-', '_'): _apply(vmin, vmax - (span*step if not isinstance(im.norm, mcolors.LogNorm) else vmax*step))
        elif e.key in ('=', '+'): _apply(vmin, vmax + (span*step if not isinstance(im.norm, mcolors.LogNorm) else vmax*step))
    _show(); fig.canvas.mpl_connect("key_press_event", _on_key); return osd

# --- Gaussian models for fitting (optional) ---
def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset
def triple_gaussian(x, a1, x0_1, sigma1, a2, x0_2, sigma2, a3, x0_3, sigma3, offset):
    return (a1*np.exp(-((x-x0_1)**2)/(2*sigma1**2)) +
            a2*np.exp(-((x-x0_2)**2)/(2*sigma2**2)) +
            a3*np.exp(-((x-x0_3)**2)/(2*sigma3**2)) + offset)

# --- 통합 GUI: 파일 선택 + 플롯 옵션 ---
def choose_files_and_options():
    """
    하나의 Tk GUI에서 파일 선택과 플롯 옵션을 함께 설정.
    반환: (file_list, opts) 또는 ([], None) if cancel.
    """
    root = tk.Tk()
    root.title("Analysis setup")
    # 창 크기 키우고 리사이즈 허용
    root.resizable(True, True)
    try:
        # 고해상도 모니터에서 폰트/위젯 스케일 살짝 키움
        if os.name == "nt":
            root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    # 레이아웃이 리사이즈에 따라 확장되도록 가중치 부여
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # 레이아웃
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")
    # 좌측 파일/우측 옵션 컬럼 및 중앙 행 확장
    frm.rowconfigure(1, weight=1)
    frm.columnconfigure(0, weight=1)  # 좌측 리스트 확장
    # 우측 옵션 영역은 자체 스크롤 패널로 처리

    # 좌: 파일 선택
    files_lbl = ttk.Label(frm, text="HDF5 files")
    files_lbl.grid(row=0, column=0, sticky="w")
    lb = tk.Listbox(frm, width=54, height=18, selectmode=tk.EXTENDED)
    lb.grid(row=1, column=0, columnspan=3, sticky="nsew")
    sb = ttk.Scrollbar(frm, orient="vertical", command=lb.yview)
    sb.grid(row=1, column=3, sticky="ns")
    lb.configure(yscrollcommand=sb.set)
    # 버튼 행도 좌측 폭에 맞춰 확장
    frm.columnconfigure(2, weight=0)

    # 파일 리스트 조작 핸들러들 추가
    def add_files():
        paths = filedialog.askopenfilenames(
            title="Select HDF5 files",
            filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*")),
            parent=root,
        )
        for p in paths:
            if p and (p not in lb.get(0, tk.END)):
                lb.insert(tk.END, p)

    def remove_sel():
        sel = list(lb.curselection())
        for i in reversed(sel):
            lb.delete(i)

    def clear_all():
        lb.delete(0, tk.END)

    ttk.Button(frm, text="Add...", command=add_files).grid(row=2, column=0, pady=(6, 10), sticky="w")
    ttk.Button(frm, text="Remove", command=remove_sel).grid(row=2, column=1, pady=(6, 10), sticky="w")
    ttk.Button(frm, text="Clear", command=clear_all).grid(row=2, column=2, pady=(6, 10), sticky="w")

    # 우: 플롯 옵션(스크롤 가능 패널)
    right = ttk.Frame(frm)
    right.grid(row=0, column=4, rowspan=3, padx=(14, 0), sticky="nsew")
    right.rowconfigure(0, weight=1)
    right.columnconfigure(0, weight=1)
    _canvas = tk.Canvas(right, highlightthickness=0)
    _vbar = ttk.Scrollbar(right, orient="vertical", command=_canvas.yview)
    _canvas.grid(row=0, column=0, sticky="nsew")
    _vbar.grid(row=0, column=1, sticky="ns")
    _canvas.configure(yscrollcommand=_vbar.set)
    opt_host = ttk.Frame(_canvas)  # 스크롤 내부 컨테이너
    _canvas.create_window((0, 0), window=opt_host, anchor="nw")
    def _on_cfg(_e):
        _canvas.configure(scrollregion=_canvas.bbox("all"))
    opt_host.bind("<Configure>", _on_cfg)
    # 마우스 휠로 스크롤
    def _wheel(e):
        try:
            _canvas.yview_scroll(-int(e.delta/120), "units")
        except Exception:
            pass
    _canvas.bind_all("<MouseWheel>", _wheel)

    # 실제 옵션 프레임은 스크롤 컨테이너(opt_host) 안에 배치
    opts_frm = ttk.LabelFrame(opt_host, text="Figures", padding=8)
    opts_frm.grid(row=0, column=0, sticky="nw")

    v_linear   = tk.BooleanVar(value=False)
    v_log      = tk.BooleanVar(value=False)
    v_dRdE     = tk.BooleanVar(value=False)
    v_dRdV     = tk.BooleanVar(value=True)
    v_d2E      = tk.BooleanVar(value=False)
    v_d2V      = tk.BooleanVar(value=False)
    v_zc_save  = tk.BooleanVar(value=True)
    v_zc_overlay = tk.BooleanVar(value=True)
    ttk.Checkbutton(opts_frm, text="Intensity (Linear)", variable=v_linear).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="Intensity (Log)",    variable=v_log).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="dR/dE",              variable=v_dRdE).grid(row=2, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="dR/dV",              variable=v_dRdV).grid(row=3, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="d2R/dE2",            variable=v_d2E).grid(row=4, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="d2R/dV2",            variable=v_d2V).grid(row=5, column=0, sticky="w")
    ttk.Separator(opts_frm, orient="horizontal").grid(row=6, column=0, sticky="ew", pady=6)
    ttk.Checkbutton(opts_frm, text="Save dR/dV zero‑crossing CSV", variable=v_zc_save).grid(row=7, column=0, sticky="w")
    ttk.Checkbutton(opts_frm, text="Overlay zero‑crossing on dR/dV", variable=v_zc_overlay).grid(row=8, column=0, sticky="w")

    # NEW: zero‑crossing 품질 필터 옵션(σ, 퍼센타일)
    zc_opts = ttk.LabelFrame(opts_frm, text="Zero‑cross filter", padding=6)
    zc_opts.grid(row=9, column=0, sticky="ew", pady=(6,0))
    v_zc_sigma = tk.DoubleVar(value=1.0)       # V축 스무딩 σ (rows 단위)
    v_zc_slope_p = tk.DoubleVar(value=80.0)    # 기울기 퍼센타일 임계값
    v_zc_con_p = tk.DoubleVar(value=60.0)      # 양쪽 크기 퍼센타일 임계값
    ttk.Label(zc_opts, text="Sigma(V-rows):").grid(row=0, column=0, sticky="w")
    ttk.Entry(zc_opts, width=6, textvariable=v_zc_sigma).grid(row=0, column=1, padx=(4,8))
    ttk.Label(zc_opts, text="Slope pct:").grid(row=0, column=2, sticky="w")
    ttk.Entry(zc_opts, width=6, textvariable=v_zc_slope_p).grid(row=0, column=3, padx=(4,8))
    ttk.Label(zc_opts, text="Contrast pct:").grid(row=0, column=4, sticky="w")
    ttk.Entry(zc_opts, width=6, textvariable=v_zc_con_p).grid(row=0, column=5, padx=(4,0))

    # NEW: Guide line 입력(전압/밀도/ν)
    guide_frm = ttk.LabelFrame(opts_frm, text="Guide lines (overlay)", padding=6)
    guide_frm.grid(row=10, column=0, sticky="ew", pady=(6,0))
    v_lines_v = tk.StringVar(value="")        # e.g., -0.8,-1.2
    v_lines_n = tk.StringVar(value="")        # e.g., -4.0,-2.0 (10^12 cm^-2)
    v_lines_nu = tk.StringVar(value="")       # e.g., 1/3, 2/3, 1
    v_n_per_nu = tk.DoubleVar(value=1.0)      # ν=1 당 밀도(10^12 cm^-2)
    ttk.Label(guide_frm, text="Voltages V:").grid(row=0, column=0, sticky="w")
    ttk.Entry(guide_frm, width=24, textvariable=v_lines_v).grid(row=0, column=1, padx=(6,0))
    ttk.Label(guide_frm, text="Densities (10^12 cm^-2):").grid(row=1, column=0, sticky="w")
    ttk.Entry(guide_frm, width=24, textvariable=v_lines_n).grid(row=1, column=1, padx=(6,0))
    ttk.Label(guide_frm, text="ν list:").grid(row=2, column=0, sticky="w")
    ttk.Entry(guide_frm, width=24, textvariable=v_lines_nu).grid(row=2, column=1, padx=(6,0))
    ttk.Label(guide_frm, text="n per ν (10^12 cm^-2):").grid(row=2, column=2, sticky="w", padx=(12,0))
    ttk.Entry(guide_frm, width=8, textvariable=v_n_per_nu).grid(row=2, column=3, padx=(4,0))

    # === Filling lines 옵션 ===
    fill_frm = ttk.LabelFrame(opts_frm, text="Filling lines (horizontal V)", padding=6)
    # 겹침 방지: guide_frm 다음 줄에 배치
    fill_frm.grid(row=11, column=0, sticky="ew", pady=(6,0))

    v_fill_overlay  = tk.BooleanVar(value=True)     # 수평선 오버레이
    v_fill_save     = tk.BooleanVar(value=True)     # CSV 저장
    v_fill_auto     = tk.BooleanVar(value=True)     # 자동(에지→클러스터)
    v_fill_binsize  = tk.DoubleVar(value=0.05)      # V 히스토그램 bin (V)
    v_fill_cov_min  = tk.DoubleVar(value=0.20)      # 최소 에너지 커버리지(0~1)
    v_fill_manual   = tk.StringVar(value="")        # 수동 입력: 예) -0.72,-1.45,-1.62

    ttk.Checkbutton(fill_frm, text="Overlay filling lines", variable=v_fill_overlay).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(fill_frm, text="Save CSV", variable=v_fill_save).grid(row=0, column=1, sticky="w")

    def _toggle_manual(*_):
        state = ("disabled" if v_fill_auto.get() else "normal")
        ent_man.configure(state=state)

    ttk.Checkbutton(fill_frm, text="Auto from dR/dV edge midpoints", variable=v_fill_auto, command=_toggle_manual).grid(row=1, column=0, columnspan=2, sticky="w")
    ttk.Label(fill_frm, text="bin (V):").grid(row=2, column=0, sticky="w")
    ttk.Entry(fill_frm, width=6, textvariable=v_fill_binsize).grid(row=2, column=1, sticky="w", padx=(4,8))
    ttk.Label(fill_frm, text="min coverage:").grid(row=2, column=2, sticky="w")
    ttk.Entry(fill_frm, width=6, textvariable=v_fill_cov_min).grid(row=2, column=3, sticky="w", padx=(4,0))

    ttk.Label(fill_frm, text="Manual V list (comma):").grid(row=3, column=0, sticky="w", pady=(6,0))
    ent_man = ttk.Entry(fill_frm, width=26, textvariable=v_fill_manual)
    ent_man.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6,0))
    _toggle_manual()

    # NEW: 파일 선택 후 기본 옵션 자동 설정
    def _auto_set_options():
        try:
            paths = list(lb.get(0, tk.END))
            if not paths:
                return
            # 첫 파일의 메타데이터로 초기화
            with h5py.File(paths[0], 'r') as h5_file:
                spectra = h5_file['spectro_data'][:, 0, :]
                wavelengths = h5_file['spectro_wavelength'][:]
                voltages = h5_file['xPositions'][:]

            voltage_min, voltage_max = float(np.min(voltages)), float(np.max(voltages))
            density_min, density_max = voltage_to_density(voltage_min), voltage_to_density(voltage_max)

            # 플롯 옵션 자동 설정: 전압 범위, 밀도 범위
            v_dRdV.set(True)  # dR/dV는 항상 표시
            v_zc_save.set(True)  # zero-crossing 저장 기본 켜기
            v_zc_overlay.set(True)  # zero-crossing 오버레이 기본 켜기
            v_fill_auto.set(True)  # filling lines 자동 설정 켜기

            # 밀도 전압 변환기준 자동 설정
            if density_min < density_max:
                v_n_per_nu.set(density_min / voltage_min)

            # NEW: 가이드라인 기본 설정 (첫 파일의 1/3, 2/3 지점)
            try:
                guide_vs = []
                with h5py.File(paths[0], 'r') as h5_file:
                    wavelengths = h5_file['spectro_wavelength'][:]
                for frac in [1/3, 2/3]:
                    E0 = 1240.0 / (wavelengths[int(len(wavelengths)*frac)])
                    guide_vs.append(E0)
                guide_vs = sorted(guide_vs)
                v_lines_v.set(",".join(f"{gv:.4f}" for gv in guide_vs))
            except Exception:
                pass

            # NEW: filling lines 기본 설정 (전압의 1/4, 1/2, 3/4 지점)
            try:
                fill_lines = []
                vmin, vmax = float(np.min(voltages)), float(np.max(voltages))
                for frac in [0.25, 0.5, 0.75]:
                    fill_lines.append(vmin + frac * (vmax - vmin))
                fill_lines = sorted(list(set(fill_lines)))  # 중복 제거 및 정렬
                v_fill_manual.set(",".join(f"{fl:.4f}" for fl in fill_lines))
            except Exception:
                pass

            messagebox.showinfo("기본 옵션 설정", "첫 파일의 메타데이터로 기본 옵션이 설정되었습니다.", parent=root)
        except Exception as e:
            messagebox.showerror("자동 설정 오류", f"기본 옵션 자동 설정 중 오류 발생:\n{e}", parent=root)

    # NEW: 파일 추가 후 자동 설정 버튼
    ttk.Button(frm, text="Auto-set options", command=_auto_set_options).grid(row=3, column=0, columnspan=5, pady=(6, 0), sticky="ew")

    # Run/Close 동작 정의
    def _run():
        files = list(lb.get(0, tk.END))
        if not files:
            messagebox.showwarning("No files", "분석할 파일을 추가하세요.", parent=root)
            return

        # 가이드 라인 입력 파싱
        guide_voltages = parse_number_list(v_lines_v.get())
        guide_densities = parse_number_list(v_lines_n.get())
        # ν 리스트를 밀도로 변환(사용자 지정 n_per_nu 배수)
        nu_list = parse_number_list(v_lines_nu.get())
        if nu_list:
            n_per_nu = float(v_n_per_nu.get())
            guide_densities += [nu * n_per_nu for nu in nu_list]

        opts = dict(
            show_linear=v_linear.get(),
            show_log=v_log.get(),
            show_dRdE=v_dRdE.get(),
            show_dRdV=v_dRdV.get(),
            show_d2E=v_d2E.get(),
            show_d2V=v_d2V.get(),
            zc_save=v_zc_save.get(),
            zc_overlay=v_zc_overlay.get(),
            zc_sigma=float(v_zc_sigma.get()),
            zc_slope_p=float(v_zc_slope_p.get()),
            zc_contrast_p=float(v_zc_con_p.get()),
            # filling lines
            fill_overlay=v_fill_overlay.get(),
            fill_save=v_fill_save.get(),
            fill_auto=v_fill_auto.get(),
            fill_binsize=float(v_fill_binsize.get()),
            fill_cov_min=float(v_fill_cov_min.get()),
            fill_manual=v_fill_manual.get().strip(),
            # guides
            guide_voltages=guide_voltages,       # [V]
            guide_densities=guide_densities,     # [10^12 cm^-2]
        )
        # 창은 닫지 않고 바로 분석 실행
        run_analysis(files, opts)

    def _cancel():
        root.destroy()

    # 하단 버튼 바도 창 폭에 맞춰 확장
    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=5, pady=(6, 0), sticky="ew")
    ttk.Button(btns, text="Run",    width=10, command=_run).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btns, text="Close",  width=10, command=_cancel).grid(row=0, column=1)

    # 창 중앙 배치
    root.update_idletasks()
    # 넉넉한 시작 크기와 최소 크기
    w, h = 1100, 700
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    root.minsize(900, 560)
    root.mainloop()

# --- 분석 본체: 기존 for-loop를 함수로 분리 ---
def run_analysis(file_list, opts):
    plt.ion()
    for file_path in file_list:
        try:
            with h5py.File(file_path, 'r') as h5_file:
                spectra = h5_file['spectro_data'][:, 0, :]
                wavelengths = h5_file['spectro_wavelength'][:]
                voltages = h5_file['xPositions'][:]

            photon_energy = 1240.0 / wavelengths  # eV (보정: 비등간격 가능)
            # 에너지/전압을 오름차순으로 정렬하고, 데이터도 그에 맞춰 열/행 재배열
            e_idx = np.argsort(photon_energy)
            E_sorted = photon_energy[e_idx]              # (Nx)
            spectra_E = spectra[:, e_idx]               # (Ny, Nx)
            # 전압 정렬(대부분 오름차순이지만 안전하게 처리)
            if not np.all(np.diff(voltages) > 0):
                v_idx = np.argsort(voltages)
                V_sorted = voltages[v_idx]
                spectra_EV = spectra_E[v_idx, :]
            else:
                V_sorted = voltages
                spectra_EV = spectra_E
            # pcolormesh용 에지 벡터
            E_edges = edges_from_centers(E_sorted)
            V_edges = edges_from_centers(V_sorted)

            v_min, v_max = 0, 5000
            energy_min, energy_max = float(E_sorted.min()), float(E_sorted.max())
            voltage_min, voltage_max = float(V_sorted.min()), float(V_sorted.max())
            density_min, density_max = voltage_to_density(voltage_min), voltage_to_density(voltage_max)

            dE_array = np.gradient(photon_energy)
            dR_dE = np.gradient(spectra, dE_array, axis=1)
            dR_dV = np.gradient(spectra, voltages, axis=0)
            d2R_dE2 = np.gradient(dR_dE, dE_array, axis=1)
            d2R_dV2 = np.gradient(dR_dV, voltages, axis=0)
            # 파생량도 동일하게 정렬
            dR_dE_sorted  = dR_dE[:, e_idx]
            dR_dV_sorted  = dR_dV[:, e_idx]
            d2R_dE2_sorted = d2R_dE2[:, e_idx]
            d2R_dV2_sorted = d2R_dV2[:, e_idx]
            if not np.all(np.diff(voltages) > 0):
                dR_dE_sorted   = dR_dE_sorted[v_idx, :]
                dR_dV_sorted   = dR_dV_sorted[v_idx, :]
                d2R_dE2_sorted = d2R_dE2_sorted[v_idx, :]
                d2R_dV2_sorted = d2R_dV2_sorted[v_idx, :]

            # dR/dV zero-crossing 추출 (급격한 전환만)
            zc_E, zc_V = [], []
            try:
                # 1) V축 스무딩
                sigma_v = float(opts.get("zc_sigma", 1.0))
                dV = np.gradient(voltages)
                dRdV_sm = gaussian_filter1d(dR_dV, sigma=sigma_v, axis=0, mode="nearest") if sigma_v > 0 else dR_dV

                # 2) 부호 전환 위치 후보
                sign_mat = np.sign(dRdV_sm)
                changes = (sign_mat[:-1, :] * sign_mat[1:, :]) < 0  # True where sign flips

                # 3) 모든 후보의 지표 계산(기울기/대비) 및 교차점 보간
                E_all, V_all, slope_all, contrast_all = [], [], [], []
                for j in range(dRdV_sm.shape[1]):   # energy index
                    flip_rows = np.where(changes[:, j])[0]
                    if flip_rows.size == 0:
                        continue
                    E0 = float(photon_energy[j])
                    for i in flip_rows:
                        y1 = float(dRdV_sm[i, j]); y2 = float(dRdV_sm[i+1, j])
                        v1 = float(voltages[i]);   v2 = float(voltages[i+1])
                        dv = max(abs(v2 - v1), 1e-12)
                        # 선형 보간 중간점(= dR/dV=0)
                        t = -y1 / (y2 - y1) if (y2 != y1) else 0.5
                        v0 = v1 + t * (v2 - v1)
                        # 지표: 급격성(기울기), 양쪽 대비(크기)
                        slope = abs(y2 - y1) / dv
                        contrast = min(abs(y1), abs(y2))
                        E_all.append(E0); V_all.append(v0)
                        slope_all.append(slope); contrast_all.append(contrast)

                if len(E_all):
                    slope_all = np.asarray(slope_all)
                    contrast_all = np.asarray(contrast_all)
                    # 4) 퍼센타일 임계값으로 필터링
                    thr_slope = np.percentile(slope_all, float(opts.get("zc_slope_p", 80.0)))
                    thr_contr = np.percentile(contrast_all, float(opts.get("zc_contrast_p", 60.0)))
                    keep = (slope_all >= thr_slope) & (contrast_all >= thr_contr)
                    zc_E = list(np.asarray(E_all)[keep])
                    zc_V = list(np.asarray(V_all)[keep])
            except Exception as _:
                pass

            # NEW: 가이드라인 전압 리스트 구성(V 직접 + 밀도→V)
            guide_vs = []
            try:
                if isinstance(opts.get("guide_voltages"), list):
                    guide_vs += [float(v) for v in opts["guide_voltages"]]
                if isinstance(opts.get("guide_densities"), list):
                    for n12 in opts["guide_densities"]:
                        v = density_to_voltage(n12)
                        if v is not None:
                            guide_vs.append(v)
                # 표시 범위 내로 제한 및 정렬
                guide_vs = sorted([v for v in guide_vs if np.isfinite(v)])
            except Exception:
                guide_vs = []

            # 공통: 가이드라인을 현재 축에 그리는 헬퍼
            def overlay_guides(ax, x_left=None, x_right=None):
                if not guide_vs:
                    return
                xl = x_left if x_left is not None else energy_min
                xr = x_right if x_right is not None else energy_max
                for gv in guide_vs:
                    if voltage_min <= gv <= voltage_max:
                        ax.axhline(gv, ls="--", lw=1.2, color="k", alpha=0.85)
                        # 우측 가장자리 근처에 전압값 라벨
                        try:
                            ax.text(
                                xr - 0.003*(xr-xl), gv, f"V={gv:.3f}",
                                va="center", ha="right", fontsize=8,
                                color="k", bbox=dict(fc="white", ec="none", alpha=0.5, pad=1.5)
                            )
                        except Exception:
                            pass

            # 공통 유틸/이벤트
            energy_sorted = E_sorted        # 스펙트럼 X축은 정렬된 에너지
            spectra_for_plot = spectra_EV   # 맵과 동일한 열 정렬(행은 전압 순서)
            selected = {"row": None}
            sel_lines = []

            def nearest_index(val, arr):
                return int(np.argmin(np.abs(np.asarray(arr) - val)))

            # NEW: 모든 맵의 선택 라인을 갱신하는 헬퍼
            def update_selected_lines(yv: float):
                for ln in sel_lines:
                    try:
                        ln.set_visible(True)
                        ln.set_ydata([yv, yv])  # axhline
                        ln.axes.figure.canvas.draw_idle()
                    except Exception:
                        pass

            def _y_on_axes(event, target_ax):
                try:
                    _, y = target_ax.transData.inverted().transform((event.x, event.y))
                    return float(y)
                except Exception:
                    return None

            def _y_on_axes_via(ax_obj, event):
                # event(x,y)를 지정한 축(ax_obj)의 데이터 좌표로 변환
                try:
                    return float(ax_obj.transData.inverted().transform((event.x, event.y))[1])
                except Exception:
                    return None

            def on_key(event):
                if selected["row"] is None:
                    return
                if event.key == "up":
                    show_spectrum_for_row(min(selected["row"] + 1, len(voltages) - 1))
                elif event.key == "down":
                    show_spectrum_for_row(max(selected["row"] - 1, 0))

            # UPDATED: 여러 축에서의 클릭을 허용하고, y좌표는 항상 primary_ax 기준으로 계산
            def on_click_factory(primary_ax):
                def _handler(event):
                    # Figure 전체 클릭 중, primary_ax의 픽셀 bbox 안인가?
                    if not primary_ax.bbox.contains(event.x, event.y):
                        return
                    # 항상 주 축 데이터 좌표로 변환
                    try:
                        _, yv = primary_ax.transData.inverted().transform((event.x, event.y))
                    except Exception:
                        return
                    # 유효 범위에서만 처리
                    if not np.isfinite(yv):
                        return
                    show_spectrum_for_row(nearest_index(yv, voltages))
                return _handler

            state = {
                "spec_fig": None, "spec_ax": None,
                "spec_y_lock": True,    # 기본 잠금 ON
                "spec_ylim": None,      # 잠금될 y-범위 저장
                "spec_ylock_txt": None  # 배지 텍스트 핸들
            }

            def show_spectrum_for_row(row_idx: int):
                # 맵 클릭 또는 키보드로 호출됨
                row_idx = int(np.clip(row_idx, 0, len(voltages)-1))
                selected["row"] = row_idx
                yv = float(voltages[row_idx])

                # NEW: 모든 맵의 선택 라인 업데이트
                update_selected_lines(yv)

                # 스펙트럼 창 표시/갱신
                if state["spec_fig"] is None or state["spec_ax"] is None:
                    fig_sp, ax_sp = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
                    state["spec_fig"], state["spec_ax"] = fig_sp, ax_sp
                    fig_sp.canvas.mpl_connect("key_press_event", on_key)
                    attach_copy_button(fig_sp, ax_sp)
                    # Y‑lock 배지 추가
                    state["spec_ylock_txt"] = attach_spec_y_lock(fig_sp, ax_sp, state)
                else:
                    # 잠금이 켜져 있으면 현재 저장된 y-lim을 유지할 준비
                    keep_ylim = state["spec_ylim"] if state.get("spec_y_lock", True) else None
                    state["spec_ax"].clear()
                    if keep_ylim is not None:
                        # clear() 후 바로 설정하면 라인 그리기 전에 적용되어 유지됨
                        try:
                            state["spec_ax"].set_ylim(*keep_ylim)
                        except Exception:
                            pass

                spec = spectra_for_plot[row_idx, :]
                ax_sp = state["spec_ax"]
                ax_sp.plot(energy_sorted, spec, color="tab:blue", lw=1.1)
                ax_sp.set_xlabel("Energy (eV)")
                ax_sp.set_ylabel("Counts")
                ax_sp.set_title(f"Spectrum @ V={yv:.4f} V (row {row_idx})")
                ax_sp.grid(alpha=0.25)
                add_top_wavelength_axis(ax_sp)

                # 잠금 로직: ON이면 저장/적용
                if state.get("spec_y_lock", True):
                    if state["spec_ylim"] is None:
                        # 첫 표시 시 현재 자동 범위를 기준으로 잠금
                        state["spec_ylim"] = ax_sp.get_ylim()
                    else:
                        # 저장된 범위를 강제 적용
                        try:
                            ax_sp.set_ylim(*state["spec_ylim"])
                        except Exception:
                            pass

                state["spec_fig"].canvas.draw_idle()

            # 선택된 플롯만 생성 (아래는 기존 코드 유지, 각 맵에 sel_line 추가/이벤트 연결)
            if opts.get("show_linear"):
                fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)
                im = ax.pcolormesh(
                    E_edges, V_edges, spectra_EV,
                    shading='auto', cmap='RdBu_r', vmin=v_min, vmax=v_max
                )
                # Linear
                ax.set_xlim(energy_min, energy_max); ax.set_ylim(voltage_min, voltage_max)
                ax.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax)
                attach_intensity_controls(fig, ax, im, data=spectra_EV)   # <- NEW
                attach_copy_button(fig, ax)

                # Linear block 끝부분에 추가
                sel_line = ax.axhline(y=0.0, color="w", ls="--", lw=1.2, alpha=0.9, visible=False)
                sel_lines.append(sel_line)
                fig.canvas.mpl_connect("button_press_event", on_click_factory(ax))
                fig.canvas.mpl_connect("key_press_event", on_key)

            if opts.get("show_log"):
                fig_log, ax_log = plt.subplots(figsize=(6, 7), constrained_layout=True)
                positive_data = spectra_EV[spectra_EV > 0]
                if positive_data.size > 0:
                    log_vmin = np.percentile(positive_data, 50)
                    log_vmax = np.percentile(positive_data, 99.5)
                else:
                    log_vmin, log_vmax = 1, 10000
                im_log = ax_log.pcolormesh(
                    E_edges, V_edges, spectra_EV,
                    shading='auto', cmap='RdBu_r',
                    norm=mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
                )
                # Log
                ax_log.set_xlim(energy_min, energy_max); ax_log.set_ylim(voltage_min, voltage_max)
                ax_log.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax_log.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax_log.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax_log)
                attach_intensity_controls(fig_log, ax_log, im_log, data=spectra_EV)  # <- NEW
                attach_copy_button(fig_log, ax_log)

                # Log block 끝부분에 추가
                sel_line_log = ax_log.axhline(y=0.0, color="w", ls="--", lw=1.0, alpha=0.9, visible=False)
                sel_lines.append(sel_line_log)
                fig_log.canvas.mpl_connect("button_press_event", on_click_factory(ax_log))
                fig_log.canvas.mpl_connect("key_press_event", on_key)

            if opts.get("show_dRdE"):
                dR_dE_flipped = dR_dE[::-1, ::-1]
                fig_deriv, ax_deriv = plt.subplots(figsize=(6, 7), constrained_layout=True)
                deriv_vmax = np.percentile(np.abs(dR_dE_flipped), 99) or 1.0
                im_deriv = ax_deriv.pcolormesh(
                    E_edges, V_edges, dR_dE_sorted,
                    shading='auto', cmap='RdBu_r',
                    norm=MidpointNormalize(vmin=-deriv_vmax, vmax=deriv_vmax, midpoint=0)
                )
                ax_deriv.set_xlim(energy_min, energy_max); ax_deriv.set_ylim(voltage_min, voltage_max)
                ax_deriv.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax_deriv.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax_deriv.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax_deriv)
                attach_intensity_controls(fig_deriv, ax_deriv, im_deriv, data=dR_dE_flipped)  # <- NEW
                # dR/dE zero-crossing 오버레이
                if opts.get("zc_overlay") and len(zc_E):
                    ax_deriv.scatter(zc_E, zc_V, s=10, c='k', marker='.', alpha=0.9, label='edge midpoints')
                    try: ax_deriv.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.3)
                    except Exception: pass
                attach_copy_button(fig_deriv, ax_deriv)

                # dR/dE block 끝부분에 추가
                sel_line_de = ax_deriv.axhline(y=0.0, color="w", ls="--", lw=1.0, alpha=0.9, visible=False)
                sel_lines.append(sel_line_de)
                fig_deriv.canvas.mpl_connect("button_press_event", on_click_factory(ax_deriv))
                fig_deriv.canvas.mpl_connect("key_press_event", on_key)

            if opts.get("show_dRdV") or opts.get("zc_overlay") or opts.get("zc_save"):
                dR_dV_flipped = dR_dV[::-1, ::-1]

            if opts.get("show_dRdV"):
                fig_dv, ax_dv = plt.subplots(figsize=(6, 7), constrained_layout=True)
                vmax_dv = np.percentile(np.abs(dR_dV_flipped), 99)
                if not np.isfinite(vmax_dv) or vmax_dv == 0:
                    vmax_dv = np.max(np.abs(dR_dV_flipped)) or 1.0
                im_dv = ax_dv.pcolormesh(
                    E_edges, V_edges, dR_dV_sorted,
                    shading='auto', cmap='RdBu_r',
                    norm=MidpointNormalize(vmin=-vmax_dv, vmax=vmax_dv, midpoint=0)
                )
                ax_dv.set_xlim(energy_min, energy_max); ax_dv.set_ylim(voltage_min, voltage_max)
                ax_dv.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax_dv.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax_dv.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax_dv)
                attach_intensity_controls(fig_dv, ax_dv, im_dv, data=dR_dV_flipped)  # <- NEW
                # NEW: zero-crossing 오버레이
                if opts.get("zc_overlay") and len(zc_E):
                    ax_dv.scatter(zc_E, zc_V, s=10, c='k', marker='.', alpha=0.9, label='edge midpoints')
                    try: ax_dv.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.3)
                    except Exception: pass
                # dR/dV block 끝부분에 추가
                sel_line_dv = ax_dv.axhline(y=0.0, color="w", ls="--", lw=1.0, alpha=0.9, visible=False)
                sel_lines.append(sel_line_dv)
                fig_dv.canvas.mpl_connect("button_press_event", on_click_factory(ax_dv))
                fig_dv.canvas.mpl_connect("key_press_event", on_key)

            if opts.get("show_d2E"):
                d2R_dE2_flipped = d2R_dE2[::-1, ::-1]
                fig_d2e, ax_d2e = plt.subplots(figsize=(6, 7), constrained_layout=True)
                vmax_d2e = np.percentile(np.abs(d2R_dE2_flipped), 99) or 1.0
                im_d2e = ax_d2e.pcolormesh(
                    E_edges, V_edges, d2R_dE2_sorted,
                    shading='auto', cmap='RdBu_r',
                    norm=MidpointNormalize(vmin=-vmax_d2e, vmax=vmax_d2e, midpoint=0)
                )
                ax_d2e.set_xlim(energy_min, energy_max); ax_d2e.set_ylim(voltage_min, voltage_max)
                ax_d2e.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax_d2e.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax_d2e.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax_d2e)
                attach_intensity_controls(fig_d2e, ax_d2e, im_d2e, data=d2R_dE2_flipped)  # <- NEW
                # d2R/dE2 block 끝부분에 추가
                sel_line_d2e = ax_d2e.axhline(y=0.0, color="w", ls="--", lw=1.0, alpha=0.9, visible=False)
                sel_lines.append(sel_line_d2e)
                fig_d2e.canvas.mpl_connect("button_press_event", on_click_factory(ax_d2e))
                fig_d2e.canvas.mpl_connect("key_press_event", on_key)

            if opts.get("show_d2V"):
                d2R_dV2_flipped = d2R_dV2[::-1, ::-1]
                fig_d2v, ax_d2v = plt.subplots(figsize=(6, 7), constrained_layout=True)
                vmax_d2v = np.percentile(np.abs(d2R_dV2_flipped), 99)
                if not np.isfinite(vmax_d2v) or vmax_d2v == 0:
                    vmax_d2v = np.max(np.abs(d2R_dV2_flipped)) or 1.0
                im_d2v = ax_d2v.pcolormesh(
                    E_edges, V_edges, d2R_dV2_sorted,
                    shading='auto', cmap='RdBu_r',
                    norm=MidpointNormalize(vmin=-vmax_d2v, vmax=vmax_d2v, midpoint=0)
                )
                ax_d2v.set_xlim(energy_min, energy_max); ax_d2v.set_ylim(voltage_min, voltage_max)
                ax_d2v.set_xlabel('Energy (eV)', fontsize=AX_LABEL_FONTSIZE)
                ax_d2v.set_ylabel('Voltage (V)', fontsize=AX_LABEL_FONTSIZE)
                ax_d2v.tick_params(labelsize=AX_TICK_FONTSIZE)
                add_top_wavelength_axis(ax_d2v)
                attach_intensity_controls(fig_d2v, ax_d2v, im_d2v, data=d2R_dV2_flipped)  # <- NEW
                # d2R/dV2 block 끝부분에 추가
                sel_line_d2v = ax_d2v.axhline(y=0.0, color="w", ls="--", lw=1.0, alpha=0.9, visible=False)
                sel_lines.append(sel_line_d2v)
                fig_d2v.canvas.mpl_connect("button_press_event", on_click_factory(ax_d2v))
                fig_d2v.canvas.mpl_connect("key_press_event", on_key)

            # save: filling lines CSV (single place)
            if opts.get("fill_save") and 'fill_lines' in locals() and fill_lines:
                import csv
                out_csv2 = os.path.join(
                    os.path.dirname(file_path),
                    os.path.splitext(os.path.basename(file_path))[0] + "_filling_lines.csv"
                )
                with open(out_csv2, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["voltage_V"])
                    for v in fill_lines:
                        w.writerow([f"{v:.9f}"])
                print(f"[FillingLines] saved: {out_csv2}")

        except Exception as e:
            print(f'An error occurred while plotting {file_path}: {e}')
    # end for file_path

# --- 진입점: GUI 먼저 띄움 (창은 닫히지 않음, Run 반복 가능) ---
if __name__ == "__main__":
    choose_files_and_options()