import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsItemGroup, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSlider
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QUrl, QRectF, QPointF, QSizeF, QSize
from PyQt6.QtGui import QBrush, QColor, QPen
import ffmpeg
import os


class DraggableRectItem(QGraphicsRectItem):
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setBrush(QBrush(QColor(255, 0, 0, 100)))
        self.setPen(QPen(Qt.GlobalColor.red, 2))
        self.offset = QPointF()
        self.resizing = False
        self.resize_handle_size = 10
        self.size_x = 0
        self.size_y = 0

    def mousePressEvent(self, event):
        rect = self.rect()
        mouse_pos = event.pos()
        if (
            abs(mouse_pos.x() - rect.right()) < self.resize_handle_size and
            abs(mouse_pos.y() - rect.bottom()) < self.resize_handle_size
        ):
            self.resizing = True
        else:
            # self.offset = event.pos() - rect.center()
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            scene_rect = self.scene().sceneRect()
            rect_width = self.rect().width()
            rect_height = self.rect().height()

            new_pos = event.pos()
            new_pos.setX(max(scene_rect.left(), min(event.scenePos().x(), scene_rect.right())) - self.x())
            new_pos.setY(max(scene_rect.top(), min(event.scenePos().y(), scene_rect.bottom())) - self.y())

            rect = self.rect()
            rect.setBottomRight(new_pos)
            self.setRect(rect)
            self.size_x = self.rect().width() / self.scene().sceneRect().width()
            self.size_y = self.rect().height() / self.scene().sceneRect().height()
        else:
            # new_center = event.scenePos() - self.offset
            new_center = event.scenePos()
            scene_rect = self.scene().sceneRect()

            rect_width = self.rect().width()
            rect_height = self.rect().height()

            new_center.setX(max(scene_rect.left() + rect_width / 2, min(new_center.x(), scene_rect.right() - rect_width / 2)))
            new_center.setY(max(scene_rect.top() + rect_height / 2, min(new_center.y(), scene_rect.bottom() - rect_height / 2)))

            self.setPos(new_center - QPointF(rect_width / 2, rect_height / 2))

        # rect = self.rect()
        # rect.setBottomRight(QPointF(self.rect().x() + self.size_x * self.scene().sceneRect().width(), self.rect().y() + self.size_y * self.scene().sceneRect().height()))
        # self.setRect(rect)

        # self.print_rect_geometry()

    def mouseReleaseEvent(self, event):
        self.resizing = False
        super().mouseReleaseEvent(event)
        # self.print_rect_geometry()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.size_x == 0 or self.size_y == 0:
            self.size_x = self.rect().width() / self.scene().sceneRect().width()
            self.size_y = self.rect().height() / self.scene().sceneRect().height()
        scene_width = self.scene().sceneRect().width()
        scene_height = self.scene().sceneRect().height()
        rect = QRectF(0, 0, self.size_x * scene_width, self.size_y * scene_height)
        handle_rect = QRectF(
            rect.right() - self.resize_handle_size,
            rect.bottom() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        painter.setBrush(QBrush(Qt.GlobalColor.blue))
        painter.drawRect(handle_rect)
        rect = self.rect()
        rect.setBottomRight(QPointF(int(rect.x() + scene_width * self.size_x), int(rect.y() + scene_height * self.size_y)))
        self.setRect(rect)

    def print_rect_geometry(self):
        """Print current coordinates and size of rectangle."""
        rect = self.rect()
        x = rect.x()
        y = rect.y()
        width = rect.width()
        height = rect.height()

        scene_pos = self.scenePos()
        scene_x = scene_pos.x()
        scene_y = scene_pos.y()

        print(f"Local Geometry: x={x}, y={y}, width={width}, height={height}")
        print(f"Scene Position: x={scene_x}, y={scene_y}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face cropper")
        self.setGeometry(100, 100, 1200, 600)

        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        self.player = QMediaPlayer(self)
        self.player.isSeekable = True
        self.player.playingChanged.connect(self.pause_button_update_text)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.player.setVideoOutput(self.video_item)
        self.media_loaded = False

        self.border_rect = QGraphicsRectItem()
        self.border_rect.setPen(QPen(QColor(0, 0, 255), 2))
        self.scene.addItem(self.border_rect)
        self.border_rect_exists = False

        self.scene.sceneRectChanged.connect(self.update_border_rect)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player.positionChanged.connect(self.on_position_changed)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(10000)
        self.progress_slider.setTickInterval(1)
        self.progress_slider.sliderPressed.connect(self.slider_mouse_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_mouse_released)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_button_clicked)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.video_cropper)
        self.start_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.view)

        self.player.stop()

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.progress_slider)
        right_layout.addWidget(self.pause_button)
        right_layout.addWidget(self.select_button)
        right_layout.addWidget(self.start_button)
        layout.addWidget(right_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.path_of_video = None

    def video_cropper(self):
        if not self.path_of_video:
            print("No video choosen.")
            return
        
        video_width = self.player.videoSink().videoSize().width()
        video_height = self.player.videoSink().videoSize().height()

        rect = self.rect_item.rect()
        rect_width = int(rect.width())
        rect_height = int(rect.height())
        rect_x = self.rect_item.pos().x()
        rect_y = self.rect_item.pos().y()

        scene_width = self.scene.width()
        scene_height = self.scene.height()
        scene_x = self.scene.sceneRect().x()
        scene_y = self.scene.sceneRect().y()

        width = int(rect_width / scene_width * video_width)
        height = int(rect_height / scene_height * video_height)
        pos_x = int((rect_x - scene_x) / scene_width * video_width)
        pos_y = int((rect_y - scene_y) / scene_height * video_height)

        print(f"({width} {height} {pos_x} {pos_y})")

        output_dir = "Images"
        (
                ffmpeg.input(self.path_of_video)
                .filter("crop", width, height, pos_x, pos_y)
                .output(os.path.join(output_dir, "Frames\\frame_%09d.jpg"), qmin=1, qscale=2)
                .run()
        )

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.BufferedMedia:
            self.media_loaded = True
            # resize video player and scene
            video_width = self.player.videoSink().videoSize().width()
            video_height = self.player.videoSink().videoSize().height()
            converted_width = int(self.height() * 0.75 * (video_width / video_height))
            converted_height = int(self.height() * 0.75)
            self.scene.setSceneRect(0, 75, converted_width, converted_height)
            self.video_item.setScale(1)
            self.video_item.setPos(0, 75)
            self.video_item.setSize(QSizeF(converted_width, converted_height))

            if not self.border_rect_exists:
                self.rect_item = DraggableRectItem(QRectF(0, 0, 200, 150))
                self.rect_item.setPos(0, 75)
                self.scene.addItem(self.rect_item)
                self.border_rect_exists = True
            # print(video_size.width() * 2.5, video_size.height() * 2.5)

    def update_border_rect(self, rect):
        self.border_rect.setRect(rect)

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Choose Video", "", "Video (*.mp4 *.webm *.mkv)"
        )
        if file:
            self.path_of_video = file
            self.player.setSource(QUrl.fromLocalFile(file))
            self.player.play()
            self.start_button.setEnabled(True)

    def on_position_changed(self, value):
        if (self.player.duration() == 0):
            self.player.stop()
            print("Error when trying to play video!")
            return
        else:
            new_value = int(value / self.player.duration() * self.progress_slider.maximum())
        self.progress_slider.setValue(new_value)

    def slider_mouse_pressed(self):
        self.player.pause()
    
    def slider_mouse_released(self):
        if self.progress_slider.value() == 0:
            new_pos = 0
        else:
            new_pos = int((self.progress_slider.value() / self.progress_slider.maximum()) * self.player.duration())
        self.player.setPosition(new_pos)
        self.player.play()

    def pause_button_clicked(self):
        if self.player.isPlaying():
            self.player.pause()
        else:
            self.player.play()

    def pause_button_update_text(self):
        if self.player.isPlaying():
            self.pause_button.setText("Pause")
        else:
            self.pause_button.setText("Play")

    def resizeEvent(self, event):
        if self.media_loaded:
            video_width = self.player.videoSink().videoSize().width()
            video_height = self.player.videoSink().videoSize().height()
            converted_width = int(self.height() * 0.75 * (video_width / video_height))
            converted_height = int(self.height() * 0.75)
            self.scene.setSceneRect(0, 75, converted_width, converted_height)
            self.video_item.setScale(1)
            self.video_item.setPos(0, 75)
            self.video_item.setSize(QSizeF(converted_width, converted_height))


if __name__ == "__main__":
    if os.name == "nt":
        os.environ["QT_MEDIA_BACKEND"] = "windows"
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())