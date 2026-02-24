#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HDF5 2D 스캔: 가능한 step size 및 range 자동 추출 도구
사용:
  python hdf5_step_range.py path/to/file.h5
  python hdf5_step_range.py            (파일 다이얼로그 등장)
옵션:
  --show-all    모든 (X,Y) 후보 조합 평가 결과 표시
  --top N       상위 N개 조합만 상세 출력 (기본 5)
  --max-cand M  좌표 1D 후보 최대 개수 (기본 12)
"""

import argparse, sys, math, statistics
from pathlib import Path
import numpy as np
import h5py

KEYWORDS = ["x", "y", "pos", "volt", "bias", "gate", "vx", "vy", "col", "row"]

def robust_step(unique_vals: np.ndarray):
    if unique_vals.size < 2:
        return 0.0, 0.0
    diffs = np.diff(unique_vals)
    # 양수 diff만
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0, 0.0
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) if diffs.size > 1 else 0.0
    spread_pct = (mad / med * 100.0) if med != 0 else 0.0
    return med, spread_pct

def score_coord_dataset(name, dset):
    score = 0
    base = name.split("/")[-1].lower()
    if dset.dtype.kind in "fiu":  # numeric
        score += 2
    if len(dset.shape) == 1 and dset.shape[0] > 1:
        score += 2
    if any(k in base for k in KEYWORDS):
        score += 3
    # 단조 증가 또는 반복패턴 + 증가
    try:
        arr = dset[()]
        if arr.ndim == 1:
            dif = np.diff(arr.astype(float))
            if np.all(dif >= 0):
                score += 1
            else:
                # 반복 후 점프 패턴 (래스터) 흔적이면 가점
                neg_ratio = np.mean(dif < 0)
                if 0 < neg_ratio < 0.3:
                    score += 1
    except Exception:
        pass
    return score

def collect_1d_candidates(hf, max_candidates=12):
    cands = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                if len(obj.shape) == 1 and obj.shape[0] > 1 and obj.shape[0] < 5_000_000:
                    cands.append((score_coord_dataset(name, obj), name, obj.shape[0]))
            except Exception:
                pass
    hf.visititems(visitor)
    cands.sort(reverse=True)
    return cands[:max_candidates]

def get_array(hf, path):
    try:
        return hf[path][()]
    except Exception:
        return None

def evaluate_pair(x_arr, y_arr, N_frames_guess=None):
    """
    x_arr, y_arr: 1D (length = frames) 또는 좌표열
    frame 수 추정: len(x_arr) == len(y_arr) == frames
    """
    info = {}
    x = np.asarray(x_arr).astype(float)
    y = np.asarray(y_arr).astype(float)
    # 고유값
    ux = np.unique(np.round(x, 12))
    uy = np.unique(np.round(y, 12))
    nx, ny = ux.size, uy.size
    med_dx, spread_x = robust_step(ux)
    med_dy, spread_y = robust_step(uy)
    info["nx"] = nx
    info["ny"] = ny
    info["x_min"] = float(ux.min()) if nx else None
    info["x_max"] = float(ux.max()) if nx else None
    info["y_min"] = float(uy.min()) if ny else None
    info["y_max"] = float(uy.max()) if ny else None
    info["x_step"] = med_dx
    info["x_step_spread_pct"] = spread_x
    info["y_step"] = med_dy
    info["y_step_spread_pct"] = spread_y
    frames_expected = nx * ny
    N_actual = x.size if x.size == y.size else None
    info["frames_expected"] = frames_expected
    info["frames_actual"] = N_actual
    if N_actual is not None:
        info["frame_mismatch"] = N_actual - frames_expected
        info["frame_mismatch_pct"] = (N_actual - frames_expected) / frames_expected * 100.0 if frames_expected else None
    else:
        info["frame_mismatch"] = None
        info["frame_mismatch_pct"] = None
    # 간단 적합 점수 (낮은 불균일, mismatch 0에 근접)
    mismatch_penalty = abs(info["frame_mismatch"]) / frames_expected if frames_expected else 1.0
    spread_penalty = (spread_x + spread_y) / 200.0  # 둘 다 %
    info["fitness"] = 1.0 / (1.0 + mismatch_penalty + spread_penalty)
    return info

def format_info(info):
    mm = info["frame_mismatch"]
    mm_pct = info["frame_mismatch_pct"]
    return (
        f" nx={info['nx']} ny={info['ny']}  "
        f"x:[{info['x_min']:.6g},{info['x_max']:.6g}] dx~{info['x_step']:.6g} (±{info['x_step_spread_pct']:.2f}%)  "
        f"y:[{info['y_min']:.6g},{info['y_max']:.6g}] dy~{info['y_step']:.6g} (±{info['y_step_spread_pct']:.2f}%)  "
        f"frames expected={info['frames_expected']} actual={info['frames_actual']} "
        f"mismatch={mm} ({mm_pct:.3f}%) fitness={info['fitness']:.4f}"
    )

def factor_pairs(n):
    res=[]
    for a in range(2, int(math.sqrt(n))+1):
        if n % a == 0:
            b = n//a
            res.append((a,b))
            if a!=b:
                res.append((b,a))
    return sorted(set(res))

def analyze_no_coords(n_frames):
    pairs = factor_pairs(n_frames)
    msg = ["[NO COORDINATE DATASETS FOUND]", f" total frames={n_frames}", " plausible (nx,ny) pairs (sorted small->large):"]
    for (a,b) in pairs[:40]:
        msg.append(f"  {a} x {b}")
    return "\n".join(msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="HDF5 파일 경로(미지정 시 파일 다이얼로그)")
    ap.add_argument("--show-all", action="store_true", help="모든 조합 상세 출력")
    ap.add_argument("--top", type=int, default=5, help="상위 N 조합 출력 (기본 5)")
    ap.add_argument("--max-cand", type=int, default=12, help="좌표 후보 최대 수 (기본 12)")
    args = ap.parse_args()

    h5_path = args.file
    if not h5_path:
        # 파일 다이얼로그 (Windows)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            h5_path = filedialog.askopenfilename(
                title="Select HDF5 file",
                filetypes=[("HDF5","*.h5 *.hdf5"),("All files","*.*")]
            )
        except Exception as e:
            print("파일 다이얼로그 실패, 경로 인자로 전달하세요:", e)
            return
    if not h5_path:
        print("파일 선택 취소.")
        return

    p = Path(h5_path)
    if not p.exists():
        print("파일 없음:", p)
        return

    print(f"[OPEN] {p}")
    with h5py.File(p, "r") as hf:
        # 후보 수집
        candidates = collect_1d_candidates(hf, max_candidates=args.max_cand)
        if not candidates:
            # 스펙트럼 shape 로 프레임 수 추정
            n_frames = None
            def find_frames(obj):
                nonlocal n_frames
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                    # frame x wavelength 가능성
                    if obj.shape[0] > 10 and obj.shape[1] > 10:
                        n_frames = obj.shape[0]
            hf.visititems(lambda n,o: find_frames(o))
            if n_frames:
                print(analyze_no_coords(n_frames))
            else:
                print("[NO DATA] 좌표/프레임 형태 추정 불가.")
            return

        print("\n[1D Coordinate Dataset Candidates]")
        for sc,name,length in candidates:
            print(f"  score={sc:2d}  len={length:6d}  {name}")

        # 실제 배열 로드
        cand_arrays=[]
        for _, name, _ in candidates:
            arr = get_array(hf, name)
            if arr is None: continue
            # NaN 제거 (복사)
            arr = np.asarray(arr)
            if arr.ndim != 1: continue
            if np.all(~np.isfinite(arr)):
                continue
            cand_arrays.append((name, arr))

        # 모든 (X,Y) 조합 평가
        results=[]
        for i,(name_x,x) in enumerate(cand_arrays):
            for j,(name_y,y) in enumerate(cand_arrays):
                if i==j: continue
                if x.shape[0] != y.shape[0]:  # 길이 다르면 같은 프레임열 아님
                    continue
                info = evaluate_pair(x,y)
                info["x_name"]=name_x
                info["y_name"]=name_y
                results.append(info)

        if not results:
            print("\n[FAIL] X,Y 조합을 구성할 동일 길이 1D 후보가 없음.")
            print("단일 좌표 배열만 있는 경우 mesh 여부 수동 확인 필요.")
            return

        # fitness 로 정렬
        results.sort(key=lambda d: d["fitness"], reverse=True)

        print("\n[TOP CANDIDATES]")
        for info in results[:args.top]:
            print(f"  X:{info['x_name']}  Y:{info['y_name']}")
            print("   "+format_info(info))

        # mismatch 없는 최우수
        perfect = [r for r in results if r["frame_mismatch"]==0]
        if perfect:
            best = perfect[0]
            print("\n[BEST PERFECT MATCH]")
            print(f"  X:{best['x_name']}  Y:{best['y_name']}")
            print("  "+format_info(best))
        else:
            best = results[0]
            print("\n[BEST (with mismatch)]")
            print(f"  X:{best['x_name']}  Y:{best['y_name']}")
            print("  "+format_info(best))

        if args.show_all:
            print("\n[ALL PAIR RESULTS]")
            for info in results:
                print(f"- X:{info['x_name']}  Y:{info['y_name']}")
                print("  "+format_info(info))

        # 추가 제안
        if best["frame_mismatch"]!=0:
            print("\n[SUGGESTION]")
            print("  프레임 수 mismatch 존재 → 마지막 미완성 row 또는 overscan 가능성.")
            print("  옵션: a) 마지막 |mismatch| 프레임 drop  b) NaN padding  c) irregular scatter 표시")

        print("\n[DONE]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT]")