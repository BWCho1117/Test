import numpy as np

# --- Doping-axis calibration ---
V0_SUM = -0.4
S_SUM  = 4.6

def v_doping(vx: float, vy: float) -> float:
    return ((vx + vy) - V0_SUM) / S_SUM

BACKGROUND = {
    "center": 1.375,
    "sigma": 0.03,
    "amp_guess": 500.0,

    # bg linewidth 제한 (eV)
    "sigma_min": 0.005,
    "sigma_max": 0.020,

    # (선택) bg 중심 고정 (흔들리면 피크를 더 먹는 경우 많음)
    "center_vary": False,

    # (선택) bg amplitude 상한도 걸고 싶으면
    # "amp_max": 400.0,
}

REGIONS = {
    # ----- HOLE side -----
    "very_low_hole": {  # -0.5 < V < 0
        "peaks": ["IX1+h", "IX2+h"],
        "centers": [1.420, 1.450],
        "sigmas":  [0.010, 0.010],
        "amps":    [800, 500],
        "constraints": {},
    },
    "low_hole": {  # -1 < V <= -0.5
        "peaks": ["IX1+h", "IX2a", "IX2+h"],
        "centers": [1.420, 1.435, 1.450],
        "sigmas":  [0.010, 0.008, 0.010],
        "amps":    [800, 200, 500],
        "constraints": {
            "IX2a_center": {"value": 1.435, "min": 1.433, "max": 1.437, "vary": True},
        },
    },
    "high_hole": {  # V <= -1
        "peaks": ["IX1+h", "IX2a", "IX2+h"],
        "centers": [1.425, 1.435, 1.460],
        "sigmas":  [0.010, 0.008, 0.010],
        "amps":    [1000, 400, 600],
        "constraints": {
            "IX2a_center": {"value": 1.435, "vary": False},
        },
    },

    # ----- GROUND -----
    "ground": {  # use this ONLY for V<=0 side
        "peaks": ["IX1", "IX2", "IX3"],
        "centers": [1.410, 1.450, 1.475],
        "sigmas":  [0.010, 0.010, 0.008],
        "amps":    [800, 900, 120],
        "constraints": {
            "IX3_center": {"min": 1.465, "max": 1.490, "vary": True},
            "IX3_sigma":  {"min": 0.004, "max": 0.015, "vary": True},
        },
    },

    # ----- ELECTRON side -----
    "low_electron": {
        "peaks": ["IX1", "IX1+e", "IX2"],
        "centers": [1.410, 1.420, 1.450],
        "sigmas":  [0.010, 0.010, 0.010],
        "amps":    [600,  200,  500],
        "order_constraints": [
            ("IX1", "IX1+e", 0.002, 0.030),
        ],
        "constraints": {
            "IX1_center":   {"min": 1.395, "max": 1.418, "vary": True},
            "IX2_center":   {"min": 1.440, "max": 1.470, "vary": True},
            "IX1_sigma":    {"min": 0.006, "max": 0.014, "vary": True},
            "IX1+e_sigma":  {"min": 0.006, "max": 0.014, "vary": True},
        },
    },
    "mid_electron": {  # 2 <= V < 2.5
        "peaks": ["IX1+e", "IX2/IX2+e", "IX'1+e", "IX'2+e"],
        "centers": [1.425, 1.450, 1.460, 1.480],
        "sigmas":  [0.010, 0.010, 0.010, 0.010],
        "amps":    [1000, 500, 300, 200],
        "constraints": {},
    },
    "high_electron": {  # 2.5 <= V < 4
        "peaks": ["IX1+e", "IX2+e"],
        "centers": [1.440, 1.460],
        "sigmas":  [0.010, 0.010],
        "amps":    [1000, 500],
        "constraints": {},
    },
    "mott": {  # V >= 4
        "peaks": ["IX1+2e"],
        "centers": [1.440],
        "sigmas":  [0.010],
        "amps":    [1000],
        "constraints": {},
    },
    "v1_electron_4peaks": {
        # p0..p3가 곧 너가 말한 에너지 순서(오름차순)
        "peaks":   ["IX1+e", "IX2+e", "IXp1+e", "IXp2+e"],
        "centers": [1.425,   1.450,   1.460,   1.480],
        "sigmas":  [0.010,   0.008,   0.008,   0.008],
        "amps":    [1200,    200,     120,     80],
        "constraints": {
            # center는 너무 빡빡하면 안 잡힐 수 있어 처음엔 ±5~10 meV 권장
            "IX1+e_center": {"min": 1.415, "max": 1.435, "vary": True},
            "IX2+e_center": {"min": 1.442, "max": 1.458, "vary": True},
            "IXp1+e_center": {"min": 1.452, "max": 1.468, "vary": True},
            "IXp2+e_center": {"min": 1.470, "max": 1.492, "vary": True},

            "IX1+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
            "IX2+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
            "IXp1+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
            "IXp2+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
        },
    },
    # --- NEW: 1<V<2에서 (IX1+e, IX2+e)만 추적 ---
    "v1to2_electron": {
        "peaks": ["IX1+e", "IX2+e"],
        "centers": [1.425, 1.450],
        "sigmas":  [0.010, 0.010],
        "amps":    [800,   400],
        "order_constraints": [
            ("IX1+e", "IX2+e", 0.005, 0.080),
        ],
        "constraints": {
            "IX1+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
            "IX2+e_sigma": {"min": 0.004, "max": 0.020, "vary": True},
        },
    },
}

def select_region(V_phys: float) -> str:
    """
    Key change:
      - for any V>0, use electron-side model that includes IX1+e.
      - ground kept for V<=0 (hole/neutral side).
    """
    eps0 = 0.05  # V≈0 근처를 ground로 취급(원하면 0.02~0.10 사이로 조절)

    if V_phys <= eps0:
        if -0.5 <= V_phys <= eps0:
            return "ground"
        if -1.0 < V_phys < -0.5:
            return "low_hole"
        return "high_hole"

    # V≈1: 4피크 anchor용
    if 0.95 <= V_phys <= 1.05:
        return "v1_electron_4peaks"

    # NEW: 1~2 구간은 IX1+e, IX2+e 추적
    if 1.05 < V_phys < 2.0:
        return "v1to2_electron"

    if V_phys < 1.05:
        return "low_electron"
    if 2.0 <= V_phys < 2.5:
        return "mid_electron"
    if 2.5 <= V_phys < 4.0:
        return "high_electron"
    return "mott"