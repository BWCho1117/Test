# DipolarIX_WS2-WSe2

Fitting and analysis codes for measured PL and DR spectra of WS2/WSe2

This repository is organized into two main parts:  
(1) Visualizing/fitting PL/DR spectra to extract exciton energies/linewidths, and  
(2) Analyzing the extracted parameters to understand excitonic properties of the WS2/WSe2 heterostructure.

## Code Description

### 1. polarization_calculation.py
**Description:**  
Loads PL spectra, wavelength, HWP angle, and magnetic field data from .hdf5 files to calculate and visualize valley polarization maps.  
**Main features:**  
- Background signal removal and cosmic ray (noise) filtering
- Wavelength-to-energy conversion and spectrum smoothing
- Grouping by HWP angle and alignment by magnetic field
- 2D map plotting for each group
- Valley polarization calculation and map visualization
- Copy all plots as PNG to clipboard

### 2. polarization_magnetic_field.py
**Description:**  
A GUI-based analysis tool for quickly visualizing magnetic field (B) sweep PL spectrum data.  
**Main features:**  
- File selection GUI
- X-axis mode selection (energy or wavelength)
- Visualization mode selection (Map/imshow or Stack)
- Automatic data loading/sorting and baseline removal
- Instant plotting in the selected mode (Map: B vs energy/wavelength, Stack: overlay spectra at multiple B with offsets)

#### Korean Version
### 1. polarization_calculation.py
#### 설명:
.hdf5 파일에서 PL 스펙트럼, 파장, HWP 각도, 자기장 데이터를 로드하여 valley polarization 맵을 계산/시각화합니다.

#### 주요 기능:
배경 신호 제거 및 cosmic ray(노이즈) 필터링
파장 → 에너지 변환, 스펙트럼 smoothing
HWP 각도별 그룹 분리 및 자기장 기준 정렬
각 그룹의 2D 맵 플롯
valley polarization 계산 및 맵 시각화
모든 플롯을 클립보드에 PNG로 복사

### 2. polarization_magnetic_field.py
#### 설명:
자기장(B) sweep PL 스펙트럼 데이터를 빠르게 시각화할 수 있는 GUI 기반 분석툴입니다.

#### 주요 기능:
파일 선택 GUI
x축 모드 선택(에너지 또는 파장)
시각화 방식 선택(Map/imshow 또는 Stack)
데이터 자동 로딩/정렬 및 baseline 제거
선택한 방식대로 즉시 플롯(Map: B vs energy/wavelength, Stack: 여러 B에서 스펙트럼을 offset으로 쌓아 그림)