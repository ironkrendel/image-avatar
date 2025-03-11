import sys
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QStackedLayout, QLabel, QVBoxLayout, QGridLayout
from PyQt6.QtMultimediaWidgets import QVideoWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video cropper")

        self.setMinimumSize(QSize(256, 144))
        self.resize(QSize(1280, 720))

        self.setStyleSheet("background-color: white")

        layout = QGridLayout()
        self.video = QVideoWidget()
        self.video.setFixedSize(QSize(int(self.size().height() * 0.8 / 9 * 16), int(self.size().height() * 0.8)))
        button = QPushButton("Test")
        button.setStyleSheet("color: black")
        button.clicked.connect(self.button_click)
        layout.addWidget(self.video, 0, 1)
        layout.addWidget(button, 1, 1)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def button_click(self):
        print("Click")

    def resizeEvent(self, event):
        if self.size().width() / 16 >= self.size().height() / 9:
            self.video.setFixedSize(QSize(int(self.size().height() * 0.8 / 9 * 16), int(self.size().height() * 0.8)))
        else:
            self.video.setFixedSize(QSize(int(self.size().width() * 0.8), int(self.size().width() * 0.8 / 16 * 9)))

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()