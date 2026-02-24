import sys
import os
import h5py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QFileDialog, QTextEdit, QLabel)
from PyQt5.QtGui import QFont

class H5MetadataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('H5 Metadata Viewer')
        self.setGeometry(150, 150, 700, 500)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)

        # 설명 레이블
        info_label = QLabel("아래 버튼을 눌러 .h5 파일이 들어있는 폴더를 선택하세요.")
        info_label.setFont(QFont('Arial', 10))
        layout.addWidget(info_label)

        # 폴더 선택 버튼
        self.select_folder_btn = QPushButton('Select Folder to Scan')
        self.select_folder_btn.setFont(QFont('Arial', 11, QFont.Bold))
        self.select_folder_btn.clicked.connect(self.open_folder_dialog)
        layout.addWidget(self.select_folder_btn)

        # 결과 표시 텍스트 영역
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont('Courier New', 10))
        self.results_text.setPlaceholderText("선택된 폴더의 .h5 파일 메타데이터가 여기에 표시됩니다.")
        layout.addWidget(self.results_text)

    def open_folder_dialog(self):
        """
        사용자가 폴더를 선택할 수 있는 대화상자를 엽니다.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder_path:
            self.results_text.clear()
            self.results_text.append(f"Scanning folder: {folder_path}\n" + "="*50 + "\n")
            self.read_metadata_in_folder(folder_path)

    def read_metadata_in_folder(self, folder_path):
        """
        지정된 폴더 내의 모든 .h5 파일에서 메타데이터를 읽어 UI에 표시합니다.
        """
        found_files = False
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".h5"):
                found_files = True
                filepath = os.path.join(folder_path, filename)
                self.results_text.append(f"📄 FILE: {filename}\n")
                try:
                    with h5py.File(filepath, 'r') as f:
                        self.print_metadata_recursively(f)
                except Exception as e:
                    self.results_text.append(f"  ERROR: Could not read file. Reason: {e}\n")
                self.results_text.append("-" * 40 + "\n")
        
        if not found_files:
            self.results_text.append("No .h5 files found in the selected folder.")

    def print_metadata_recursively(self, h5_object):
        """
        H5 객체를 재귀적으로 방문하여 모든 메타데이터를 UI에 표시합니다.
        """
        # 파일 루트의 속성 먼저 처리
        if h5_object.attrs:
            self.results_text.append("  ▶ Path: / (Root Attributes)")
            for key, val in h5_object.attrs.items():
                self.results_text.append(f"    - {key}: {val}")
        
        # 파일 내의 모든 그룹/데이터셋 방문
        def visit_func(name, obj):
            if obj.attrs:
                self.results_text.append(f"  ▶ Path: /{name}")
                for key, val in obj.attrs.items():
                    self.results_text.append(f"    - {key}: {val}")
        
        h5_object.visititems(visit_func)

if __name__ == '__main__':
    # h5py 라이브러리가 설치되어 있는지 확인
    try:
        import h5py
    except ImportError:
        print("Error: h5py is not installed. Please install it using 'pip install h5py'")
        sys.exit(1)

    app = QApplication(sys.argv)
    ex = H5MetadataViewer()
    ex.show()
    sys.exit(app.exec_())
