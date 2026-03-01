# DipolarIX_WS2-WSe2

Fitting and analysis codes for measured PL and DR spectra of WS2/WSe2

This repository is organized into two main parts:  
(1) Visualizing/fitting PL/DR spectra to extract exciton energies/linewidths, and  
(2) Analyzing the extracted parameters to understand excitonic properties of the WS2/WSe2 heterostructure.

## Code Description

### 2. Visualization
### 2-1. gate_sweep_full.py
**Description:**  
Loads spectrum data from .spe files and visualizes a 2D colormap based on gate X, Y coordinates.  
**Main features:**  
- Click on a specific position to plot the corresponding spectrum (Intensity vs. energy)
- Visualize linecuts (stack/map) from selected regions

### 2-2. gate_sweep_ratio.py
**Description:**  
Loads doping/e-field sweep data from hdf5 files for various visualization and analysis methods.  
**Main features:**  
- Automatically generates maps of intensity (linear/log), 1st/2nd derivatives (dR/dE, dR/dV, d²R/dE², d²R/dV²)
- Click on any map to instantly plot the spectrum at that voltage
- Zero-crossing (edge) detection, guideline/filling line overlay, and CSV export



---
#### Korean Version
## 코드 설명
### 2. Visualization
### 2-1. gate_sweep_full.py
#### 설명:
.spe 파일에서 스펙트럼 데이터를 불러와 gate X, Y 좌표 기준으로 2D 컬러맵을 시각화합니다.
주요 기능:
특정 위치 클릭 시 해당 스펙트럼(Intensity vs. energy) 플롯
선택 영역의 linecut(stack/map) 시각화

#### 주요 기능:
특정 위치 클릭 시 해당 스펙트럼(Intensity vs. energy) 플롯
선택 영역의 linecut(stack/map) 시각화

### 2-2. gate_sweep_ratio.py
#### 설명:
hdf5 파일에서 doping/e-field sweep 데이터를 불러와 다양한 방식으로 시각화 및 분석합니다.

#### 주요 기능:
Intensity(선형/로그), 1차/2차 도함수(dR/dE, dR/dV, d²R/dE², d²R/dV²) 맵 자동 생성
각 맵에서 클릭 시 해당 전압의 스펙트럼을 바로 플롯
zero-crossing(에지) 검출, 가이드라인/필링라인 오버레이 및 CSV 저장