# DipolarIX_WS2-WSe2
Fitting and analysis codes for measured PL and DR spectra of WS2/WSe2

This repository is organized into two main parts:
(1) visualizing/ fitting PL/DR spectra to extract exciton energies/linewidths, and
(2) analyzing the extracted parameters to understand excitonic properties of the WS2/WSe2 heterostructure.

## 코드 설명

### 1. gate_sweep_full.py
#### 설명:
.spe 파일에서 스펙트럼 데이터를 불러와 gate X, Y 좌표 기준으로 2D 컬러맵을 시각화합니다.
주요 기능:
특정 위치 클릭 시 해당 스펙트럼(Intensity vs. energy) 플롯
선택 영역의 linecut(stack/map) 시각화

#### 주요 기능:
특정 위치 클릭 시 해당 스펙트럼(Intensity vs. energy) 플롯
선택 영역의 linecut(stack/map) 시각화

### 2. gate_sweep_ratio.py
#### 설명:
hdf5 파일에서 doping/e-field sweep 데이터를 불러와 다양한 방식으로 시각화 및 분석합니다.

#### 주요 기능:
Intensity(선형/로그), 1차/2차 도함수(dR/dE, dR/dV, d²R/dE², d²R/dV²) 맵 자동 생성
각 맵에서 클릭 시 해당 전압의 스펙트럼을 바로 플롯
zero-crossing(에지) 검출, 가이드라인/필링라인 오버레이 및 CSV 저장

### 3. polarization_calculation.py
#### 설명:
.hdf5 파일에서 PL 스펙트럼, 파장, HWP 각도, 자기장 데이터를 로드하여 valley polarization 맵을 계산/시각화합니다.

#### 주요 기능:
배경 신호 제거 및 cosmic ray(노이즈) 필터링
파장 → 에너지 변환, 스펙트럼 smoothing
HWP 각도별 그룹 분리 및 자기장 기준 정렬
각 그룹의 2D 맵 플롯
valley polarization 계산 및 맵 시각화
모든 플롯을 클립보드에 PNG로 복사

### 4. polarization_magnetic_field.py
#### 설명:
자기장(B) sweep PL 스펙트럼 데이터를 빠르게 시각화할 수 있는 GUI 기반 분석툴입니다.

#### 주요 기능:
파일 선택 GUI
x축 모드 선택(에너지 또는 파장)
시각화 방식 선택(Map/imshow 또는 Stack)
데이터 자동 로딩/정렬 및 baseline 제거
선택한 방식대로 즉시 플롯(Map: B vs energy/wavelength, Stack: 여러 B에서 스펙트럼을 offset으로 쌓아 그림)

## 폴더 구조 예시



