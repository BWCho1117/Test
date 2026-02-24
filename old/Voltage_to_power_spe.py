import os
import re
import shutil
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

SLOPE = 458548.7349 ##291923.1697 for low 106 to 107// 306439.2901 for high 106 to 104
INTERCEPT = 0 ##13.9899 for low 106 to 107// -23.2072 for high 106 to 104
BG = 0 

def parse_filename(fname):
    match = re.match(r"([\d\.]+)s_(\d+)_(\d+)_([\d\.]+)(mV|V)\.spe", fname)
    if not match:
        return None
    time = match.group(1)
    amp = match.group(2)
    amp_exp = match.group(3)
    voltage = float(match.group(4))
    unit = match.group(5)
    if unit == "V":
        voltage = voltage * 1000
    # Fix: interpret "10_3" as 10**3 = 1000
    try:
        exp = int(amp_exp)
        if amp == '10':
            gain = 10 ** exp
        else:
            gain = float(amp) * (10 ** exp)
    except Exception:
        # fallback to previous behavior
        gain = float(f"{amp}e{amp_exp}")
    return time, gain, voltage

def calc_power(voltage, gain):
    power = SLOPE * (voltage / gain) + INTERCEPT
    adjusted = power - BG
    return round(adjusted, 4)  # 소수점 네째자리까지 반영

def main():
    app = QApplication([])
    # 1. 폴더 선택
    folder = QFileDialog.getExistingDirectory(None, "폴더 선택")
    if not folder:
        QMessageBox.warning(None, "오류", "폴더를 선택하지 않았습니다.")
        return

    # 2. 폴더 내 spe 파일 리스트
    spe_files = [f for f in os.listdir(folder) if f.endswith(".spe")]
    if not spe_files:
        QMessageBox.warning(None, "오류", "폴더에 .spe 파일이 없습니다.")
        return

    # 3. spe 파일들 중에서 직접 선택
    files, _ = QFileDialog.getOpenFileNames(
        None, "변환할 .spe 파일 선택", folder, "SPE Files (*.spe)"
    )
    if not files:
        QMessageBox.warning(None, "오류", "파일을 선택하지 않았습니다.")
        return

    # 4. 변환 폴더 생성
    new_folder = os.path.join(folder, "converted_spe")
    os.makedirs(new_folder, exist_ok=True)

    # 5. 선택한 파일만 변환
    for fpath in files:
        fname = os.path.basename(fpath)
        params = parse_filename(fname)
        if params:
            time, gain, voltage = params
            adj_power = calc_power(voltage, gain)
            # sample에 저장되는 값 및 파일명은 계산 결과의 절반을 사용
            sample_power = round(adj_power / 2.0, 4)
            new_name = f"{sample_power}nW_{time}s.spe"
        else:
            new_name = fname
        shutil.copy2(fpath, os.path.join(new_folder, new_name))

    QMessageBox.information(None, "완료", f"변환된 파일이\n{new_folder}\n에 저장되었습니다.")

if __name__ == "__main__":
    main()