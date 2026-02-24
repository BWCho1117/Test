from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTableWidget, QTableWidgetItem, QLineEdit, QLabel, QPushButton, 
                             QGroupBox, QHeaderView, QMessageBox, QComboBox, QCheckBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

class PowerCalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Power Calibration Tool')
        self.setGeometry(100, 100, 900, 700)

        # 기본 데이터
        self.voltages = [
            54.45, 8.80, 3.72, 5.58, 6.85, 11.70, 17.2, 24.4, 29.9, 38.2, 45.2, 65.2, 73.1, 82.1, 84.2,
            115.2, 122.2, 145.1, 157.5, 207.0, 251.5, 296.8, 370.4, 463.5, 549.4, 725.5, 772.2, 893.9
        ]
        self.powers = [
            133.0, 34.0, 22.3, 27.2, 30.0, 40.4, 52.8, 68.0, 79.4, 97.9, 115.9, 136.5, 175.3, 195.5, 196.0,
            266.1, 281.8, 319.4, 358.5, 450.5, 556.7, 660.8, 820.8, 1024.0, 1205.0, 1573.0, 1712.0, 1961.0
        ]
        self.gains = ['1e5'] * len(self.voltages)  # 기본값 10^5

        self.slope = 0
        self.intercept = 0

        self.init_ui()
        self.populate_table()
        self.perform_fit_and_plot()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left side layout
        left_layout = QVBoxLayout()

        # Data Table
        data_group = QGroupBox("Calibration Data")
        data_layout = QVBoxLayout(data_group)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Voltage (mV)', 'Power (nW)', 'Gain'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        data_layout.addWidget(self.table)

        # Buttons for table manipulation
        button_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("Add New Row")
        self.remove_point_btn = QPushButton("Remove Selected Row")
        self.save_csv_btn = QPushButton("Save to CSV")
        self.load_csv_btn = QPushButton("Load from CSV")
        button_layout.addWidget(self.add_row_btn)
        button_layout.addWidget(self.remove_point_btn)
        button_layout.addWidget(self.save_csv_btn)
        button_layout.addWidget(self.load_csv_btn)
        data_layout.addLayout(button_layout)
        left_layout.addWidget(data_group)

        # Calculator
        calculator_group = QGroupBox("Power Calculator")
        calculator_layout = QVBoxLayout(calculator_group)
        self.fit_label = QLabel("Fit: Power = m * (V/Gain) + c")
        self.fit_label.setFont(QFont('Arial', 10, QFont.Bold))
        calculator_layout.addWidget(self.fit_label)

        hbox_voltage = QHBoxLayout()
        self.voltage_label = QLabel("Enter Voltage (mV):")
        hbox_voltage.addWidget(self.voltage_label)
        self.voltage_input = QLineEdit()
        self.voltage_input.setPlaceholderText("e.g., 1.23")
        hbox_voltage.addWidget(self.voltage_input)
        calculator_layout.addLayout(hbox_voltage)

        hbox_gain = QHBoxLayout()
        hbox_gain.addWidget(QLabel("Gain:"))
        self.gain_input = QComboBox()
        self.gain_input.addItems(["1e3", "1e4", "1e5", "1e6", "1e7"])  # 1e3, 1e4 추가
        calculator_layout.addWidget(self.gain_input)
        hbox_gain.addStretch()
        calculator_layout.addLayout(hbox_gain)

        hbox_power = QHBoxLayout()
        hbox_power.addWidget(QLabel("Calculated Power (μW):"))  # nW → μW
        self.power_output = QLabel("")
        self.power_output.setFont(QFont('Arial', 10, QFont.Bold))
        self.power_output.setStyleSheet("color: blue;")
        hbox_power.addWidget(self.power_output)
        hbox_power.addStretch()
        calculator_layout.addLayout(hbox_power)

        # Background power input
        hbox_background = QHBoxLayout()
        hbox_background.addWidget(QLabel("Background Power (nW):"))
        self.background_power_input = QLineEdit()
        self.background_power_input.setPlaceholderText("e.g., 10.5")
        hbox_background.addWidget(self.background_power_input)
        hbox_background.addStretch()
        calculator_layout.addLayout(hbox_background)

        # Fit options
        self.fit_origin_cb = QCheckBox("Subtract background and fit through origin")
        self.fit_origin_cb.setChecked(True)
        calculator_layout.addWidget(self.fit_origin_cb)
        self.log_axes_cb = QCheckBox("Use log axes on plot")
        self.log_axes_cb.setChecked(False)
        calculator_layout.addWidget(self.log_axes_cb)

        # Calculated power label
        self.calculated_power_label = QLabel("Calculated Power:")
        self.calculated_power_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.calculated_power_label.setStyleSheet("color: green;")
        calculator_layout.addWidget(self.calculated_power_label)

        # Adjusted power label (after subtracting background)
        self.adjusted_power_label = QLabel("Adjusted Power:")
        self.adjusted_power_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.adjusted_power_label.setStyleSheet("color: orange;")
        calculator_layout.addWidget(self.adjusted_power_label)

        # Sample power label (Adjusted Power / 2)
        self.sample_power_label = QLabel("Sample Power:")
        self.sample_power_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.sample_power_label.setStyleSheet("color: purple;")
        calculator_layout.addWidget(self.sample_power_label)

        # Unit selection for power
        self.unit_input = QComboBox()
        self.unit_input.addItems(["nW", "uW"])
        calculator_layout.addWidget(QLabel("Power Unit:"))
        calculator_layout.addWidget(self.unit_input)

        left_layout.addWidget(calculator_group)
        left_layout.addStretch()
        main_layout.addLayout(left_layout, 1)

        # Right side: Plot
        plot_group = QGroupBox("Linear Fit Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_group, 2)

        # Connections
        self.voltage_input.textChanged.connect(self.calculate_power)
        self.gain_input.currentTextChanged.connect(self.calculate_power)
        self.background_power_input.textChanged.connect(self.perform_fit_and_plot)
        self.fit_origin_cb.stateChanged.connect(self.perform_fit_and_plot)
        self.log_axes_cb.stateChanged.connect(self.perform_fit_and_plot)
        self.add_row_btn.clicked.connect(self.add_new_row)
        self.remove_point_btn.clicked.connect(self.remove_data_point)
        self.table.itemChanged.connect(self.update_data_from_table)
        self.save_csv_btn.clicked.connect(self.save_to_csv)
        self.load_csv_btn.clicked.connect(self.load_from_csv)

    def populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.voltages))
        gain_values = ['1e3', '1e4', '1e5']
        for i, (v, p, g) in enumerate(zip(self.voltages, self.powers, self.gains)):
            self.table.setItem(i, 0, QTableWidgetItem(str(v)))
            self.table.setItem(i, 1, QTableWidgetItem(str(p)))
            gain_combo = QComboBox()
            gain_combo.addItems(gain_values)
            gain_combo.setEditable(True)
            gain_combo.setCurrentText(g)
            gain_combo.currentTextChanged.connect(lambda _, row=i: self.update_gain_from_table(row))
            self.table.setCellWidget(i, 2, gain_combo)
        self.table.blockSignals(False)

    def update_gain_from_table(self, row):
        gain_widget = self.table.cellWidget(row, 2)
        if gain_widget:
            self.gains[row] = gain_widget.currentText()
            self.perform_fit_and_plot()

    def perform_fit_and_plot(self):
        # (Voltage / Gain)로 변환해서 fitting
        voltages_for_fit = []
        valid_powers = []
        for v, g, p in zip(self.voltages, self.gains, self.powers):
            try:
                gain_val = float(g)
                voltages_for_fit.append(float(v) / gain_val)
                valid_powers.append(float(p))
            except (ValueError, TypeError, ZeroDivisionError):
                continue

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if len(voltages_for_fit) > 1:
            x = np.asarray(voltages_for_fit, dtype=float)
            y = np.asarray(valid_powers, dtype=float)

            # 배경
            try:
                bg = float(self.background_power_input.text())
            except Exception:
                bg = 0.0

            if self.fit_origin_cb.isChecked():
                # (y - bg) 를 원점 통과 직선으로 피팅: slope = (x·(y-bg)) / (x·x)
                y_adj = y - bg
                denom = np.dot(x, x)
                if denom == 0:
                    self.slope, self.intercept = 0.0, bg
                else:
                    self.slope = float(np.dot(x, y_adj) / denom)
                    self.intercept = bg  # 원래 단위로는 bg를 절편으로 표시
                sign = "+" if self.intercept >= 0 else "-"
                self.fit_label.setText(f"Power = {self.slope:.4f} * (V/Gain) {sign} {abs(self.intercept):.4f}  [bg used]")
            else:
                # 일반 최소제곱 직선 피팅
                self.slope, self.intercept = np.polyfit(x, y, 1)
                sign = "+" if self.intercept >= 0 else "-"
                self.fit_label.setText(f"Power = {self.slope:.4f} * (V/Gain) {sign} {abs(self.intercept):.4f}")

            # 산점도
            ax.scatter(x, y, label='Data Points', color='red', zorder=5)

            # 피팅선
            x_fit = np.linspace(max(min(x), 1e-12), max(x), 200)
            y_fit = self.slope * x_fit + self.intercept
            ax.plot(x_fit, y_fit, label=f'Linear Fit (m={self.slope:.2f})', color='blue', zorder=10)
        else:
            self.slope, self.intercept = 0.0, 0.0
            self.fit_label.setText("Fit: Not enough data")

        # 축 스케일
        if self.log_axes_cb.isChecked():
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        ax.set_xlabel('Voltage / Gain (mV)')
        ax.set_ylabel('Power (nW)')
        ax.set_title('Voltage/Gain vs. Power')

        ax.legend()
        ax.grid(True, which='both', ls=':', alpha=0.6)
        self.canvas.draw()
        self.calculate_power()

    def calculate_power(self):
        gain = float(self.gain_input.currentText())
        unit = self.unit_input.currentText()
        try:
            voltage = float(self.voltage_input.text())
        except Exception:
            voltage = 0.0

        # 피팅 결과 (nW 단위)
        calculated_power_nw = self.slope * (voltage / gain) + self.intercept

        # background 값 읽기 (nW 단위 입력)
        try:
            background_nw = float(self.background_power_input.text())
        except Exception:
            background_nw = 0.0

        # Adjusted Power 계산 (nW 단위)
        adjusted_power_nw = calculated_power_nw - background_nw

        # Sample Power (Adjusted Power / 2)
        sample_power_nw = adjusted_power_nw / 2

        # 단위 변환 및 표시
        if unit == "uW":
            calc_display = calculated_power_nw / 1000.0
            adj_display = adjusted_power_nw / 1000.0
            sample_display = sample_power_nw / 1000.0
            unit_label = "μW"
        else:  # "nW"
            calc_display = calculated_power_nw
            adj_display = adjusted_power_nw
            sample_display = sample_power_nw
            unit_label = "nW"

        self.calculated_power_label.setText(f"Calculated Power: {calc_display:.6f} {unit_label}")
        self.adjusted_power_label.setText(f"Adjusted Power: {adj_display:.6f} {unit_label}")
        self.sample_power_label.setText(f"Sample Power: {sample_display:.6f} {unit_label}")

    def add_new_row(self):
        self.table.blockSignals(True)
        rowCount = self.table.rowCount()
        self.table.insertRow(rowCount)
        self.voltages.append(0.0)
        self.powers.append(0.0)
        self.gains.append('1e5')
        self.table.setItem(rowCount, 0, QTableWidgetItem("0.0"))
        self.table.setItem(rowCount, 1, QTableWidgetItem("0.0"))
        gain_combo = QComboBox()
        gain_combo.addItems(['1e3', '1e4', '1e5'])
        gain_combo.setEditable(True)
        gain_combo.setCurrentText('1e5')
        gain_combo.currentTextChanged.connect(lambda _, row=rowCount: self.update_gain_from_table(row))
        self.table.setCellWidget(rowCount, 2, gain_combo)
        self.table.blockSignals(False)
        self.perform_fit_and_plot()

    def remove_data_point(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "Please select a row to remove.")
            return
        for index in sorted([r.row() for r in selected_rows], reverse=True):
            self.table.removeRow(index)
            del self.voltages[index]
            del self.powers[index]
            del self.gains[index]
        self.perform_fit_and_plot()

    def update_data_from_table(self, item):
        row = item.row()
        col = item.column()
        new_text = item.text()
        try:
            new_value = float(new_text)
            if col == 0: # Voltage
                self.voltages[row] = new_value
            elif col == 1: # Power
                self.powers[row] = new_value
            self.perform_fit_and_plot()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid number.")
            self.table.blockSignals(True)
            old_value = self.voltages[row] if col == 0 else self.powers[row]
            item.setText(str(old_value))
            self.table.blockSignals(False)

    def save_to_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        import csv
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            # 파일 확장자 자동 추가
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Voltage (mV)', 'Power (nW)', 'Gain'])
                    for v, p, g in zip(self.voltages, self.powers, self.gains):
                        writer.writerow([v, p, g])
                QMessageBox.information(self, "Saved", f"Saved to:\n{filename}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save file:\n{e}")

    def load_from_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        import csv
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            try:
                voltages, powers, gains = [], [], []
                with open(filename, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        voltages.append(float(row['Voltage (mV)']))
                        powers.append(float(row['Power (nW)']))
                        gains.append(row['Gain'])
                self.voltages, self.powers, self.gains = voltages, powers, gains
                self.populate_table()
                self.perform_fit_and_plot()
                QMessageBox.information(self, "Loaded", f"Loaded from:\n{filename}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load file:\n{e}")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = PowerCalibrationApp()
    win.show()
    sys.exit(app.exec_())
