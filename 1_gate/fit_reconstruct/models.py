import numpy as np
from lmfit import Model, Parameters

def gaussian(x, amp, cen, sig):
    return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)

def build_model(region_cfg: dict, bg_cfg: dict, *, include_background: bool = True):
    """
    Build lmfit model for a region.
    include_background=False면 bg_ 가우시안 항을 모델에서 제거.
    """
    model = None

    if include_background:
        model = Model(gaussian, prefix="bg_")
    # peaks
    for i in range(len(region_cfg["peaks"])):
        peak_model = Model(gaussian, prefix=f"p{i}_")
        model = peak_model if model is None else (model + peak_model)

    params = Parameters()

    # background params
    if include_background:
        params.add("bg_amp", value=bg_cfg["amp_guess"], min=0)
        params.add("bg_cen", value=bg_cfg["center"], vary=False)
        params.add("bg_sig", value=bg_cfg["sigma"], min=0.015, max=0.06)

    # peaks params
    for i, name in enumerate(region_cfg["peaks"]):
        params.add(f"p{i}_amp", value=float(region_cfg["amps"][i]), min=0)
        params.add(f"p{i}_cen", value=float(region_cfg["centers"][i]))
        params.add(f"p{i}_sig", value=float(region_cfg["sigmas"][i]), min=0.004, max=0.03)

        # center constraints (existing)
        key_c = f"{name}_center"
        c = region_cfg.get("constraints", {}).get(key_c)
        if c:
            if "value" in c: params[f"p{i}_cen"].value = c["value"]
            if "min" in c:   params[f"p{i}_cen"].min = c["min"]
            if "max" in c:   params[f"p{i}_cen"].max = c["max"]
            if "vary" in c:  params[f"p{i}_cen"].vary = c["vary"]

        # --- NEW: sigma constraints ---
        key_s = f"{name}_sigma"
        s = region_cfg.get("constraints", {}).get(key_s)
        if s:
            if "value" in s: params[f"p{i}_sig"].value = s["value"]
            if "min" in s:   params[f"p{i}_sig"].min = s["min"]
            if "max" in s:   params[f"p{i}_sig"].max = s["max"]
            if "vary" in s:  params[f"p{i}_sig"].vary = s["vary"]

    # --- Optional: enforce peak ordering via delta parameters ---
    # region_cfg["order_constraints"] = [("IX1", "IX1+e", 0.002, 0.03), ...]
    name_to_i = {name: i for i, name in enumerate(region_cfg.get("peaks", []))}
    for a, b, dmin, dmax in region_cfg.get("order_constraints", []):
        if a not in name_to_i or b not in name_to_i:
            continue
        ia = name_to_i[a]
        ib = name_to_i[b]
        dname = f"d_{ia}_{ib}"
        if dname not in params:
            params.add(dname, value=float(dmin) * 2, min=float(dmin), max=float(dmax))
        # p{ib}_cen is no longer free; it's expressed as p{ia}_cen + delta
        params[f"p{ib}_cen"].expr = f"p{ia}_cen + {dname}"

    return model, params

def seed_params(params: Parameters, seed: dict, *, only_prefixes: tuple[str, ...] = ("bg_", "p", "d_")):
    """
    Warm-start helper: copy numeric values from a previous fit into new params.
    - seed: dict(name -> value) e.g. {"p0_cen": 1.42, "d_0_1": 0.01, ...}
    - only_prefixes: which parameter name prefixes to accept
    """
    if not seed:
        return params

    for k, v in seed.items():
        if not isinstance(k, str):
            continue
        if not k.startswith(only_prefixes):
            continue
        if k in params:
            try:
                params[k].set(value=float(v))
            except Exception:
                pass

    # expr 파라미터는 내부적으로 업데이트가 필요할 수 있음
    if hasattr(params, "update_constraints"):
        params.update_constraints()

    return params