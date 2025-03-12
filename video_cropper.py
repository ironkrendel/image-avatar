import sys
from PyQt6.QtGui import QPixmap, QPainter, QColor, QCursor
from PyQt6.QtCore import QSize, Qt, QUrl, QRect
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QStackedLayout, QLabel, QVBoxLayout, QGridLayout
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtMultimedia import QMediaPlayer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video cropper")

        self.setMinimumSize(QSize(256, 144))
        self.resize(QSize(1280, 720))

        self.setStyleSheet("background-color: white")

        gridLayout = QGridLayout()
        self.layout = QStackedLayout()
        self.player = QMediaPlayer()

        self.overlay = OverlayWidget(self)
        self.overlay.setGeometry(50, 50, 300, 200)
        self.layout.addWidget(self.overlay)
        # self.overlay.show()

        self.video = QVideoWidget()
        self.video.setFixedSize(QSize(int(self.size().height() * 0.8 / 9 * 16), int(self.size().height() * 0.8)))

        select_file_button = QPushButton("Select video")
        button = QPushButton("Start")

        button.setStyleSheet("color: black")
        select_file_button.setStyleSheet("color: black")

        select_file_button.clicked.connect(self.get_file_location)
        button.clicked.connect(self.button_click)

        gridLayout.addWidget(self.video, 0, 1)
        gridLayout.addWidget(button, 1, 1)
        gridLayout.addWidget(select_file_button, 2,1)
        
        gridWidget = QWidget()
        gridWidget.setLayout(gridLayout)
        self.layout.addWidget(gridWidget)

        widget = QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)

    def button_click(self):
        print("Start")


    def get_file_location(self):
        dialog = QFileDialog(self)
        dialog.setNameFilter(".mp4")
        file = dialog.getOpenFileName(self, "Teto", "./", "Video Files (*.mp4;*.webm)")
        self.player.setSource(QUrl(file[0]))
        self.player.setVideoOutput(self.video)
        self.video.show()
        self.player.play()

    def resizeEvent(self, event):
        if self.size().width() / 16 >= self.size().height() / 9:
            self.video.setFixedSize(QSize(int(self.size().height() * 0.8 / 9 * 16), int(self.size().height() * 0.8)))
        else:
            self.video.setFixedSize(QSize(int(self.size().width() * 0.8), int(self.size().width() * 0.8 / 16 * 9)))


class OverlayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(QColor(0, 0, 0, 128))
        painter.drawRect(self.rect())

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()