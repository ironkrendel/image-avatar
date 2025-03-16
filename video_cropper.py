import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QUrl, QRectF, QPointF
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

    def mousePressEvent(self, event):
        rect = self.rect()
        mouse_pos = event.pos()
        if (
            abs(mouse_pos.x() - rect.right()) < self.resize_handle_size and
            abs(mouse_pos.y() - rect.bottom()) < self.resize_handle_size
        ):
            self.resizing = True
        else:
            self.offset = event.pos() - rect.center()
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            new_pos = event.pos()
            rect = self.rect()
            rect.setBottomRight(new_pos)
            self.setRect(rect)
        else:
            new_center = event.scenePos() - self.offset
            scene_rect = self.scene().sceneRect()

            rect_width = self.rect().width()
            rect_height = self.rect().height()

            new_center.setX(max(scene_rect.left() + rect_width / 2, min(new_center.x(), scene_rect.right() - rect_width / 2)))
            new_center.setY(max(scene_rect.top() + rect_height / 2, min(new_center.y(), scene_rect.bottom() - rect_height / 2)))

            self.setPos(new_center - QPointF(rect_width / 2, rect_height / 2))

        self.print_rect_geometry()

    def mouseReleaseEvent(self, event):
        self.resizing = False
        super().mouseReleaseEvent(event)
        self.print_rect_geometry()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        rect = self.rect()
        handle_rect = QRectF(
            rect.right() - self.resize_handle_size,
            rect.bottom() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        painter.setBrush(QBrush(Qt.GlobalColor.blue))
        painter.drawRect(handle_rect)

    def print_rect_geometry(self):
        """Выводит текущие координаты и размеры прямоугольника."""
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
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.player.setVideoOutput(self.video_item)

        self.border_rect = QGraphicsRectItem()
        self.border_rect.setPen(QPen(QColor(0, 0, 255), 2))
        self.scene.addItem(self.border_rect)

        self.scene.sceneRectChanged.connect(self.update_border_rect)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

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
        right_layout.addWidget(self.select_button)
        right_layout.addWidget(self.start_button)
        layout.addWidget(right_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.path_of_video = None

    def video_cropper(self):
        if not self.path_of_video:
            print("Видео не выбрано.")
            return


        rect = self.rect_item.rect()
        width = int(rect.width()) * 1.6
        height = int(rect.height()) * 1.2


        scene_pos = self.rect_item.scenePos()
        scene_x = int(scene_pos.x()) * 2.5
        scene_y = int(scene_pos.y()) * 2.5

        print("(",width, height, scene_x, scene_y,")")


        output_dir = "Images"

        (
                ffmpeg.input(self.path_of_video)
                .filter("crop", width, height, scene_x, scene_y)
                .output(os.path.join(output_dir, "frame_%04d.png"), vframes=300)
                .run()
        )

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.video_item.setScale(2.5)
            video_size = self.video_item.size()
            self.scene.setSceneRect(0, 75, video_size.width() * 2.5, 450)
            self.video_item.setPos(0, 0)

            self.rect_item = DraggableRectItem(QRectF(0, 0, 200, 150))
            self.scene.addItem(self.rect_item)
            print(video_size.width() * 2.5, video_size.height() * 2.5)

    def update_border_rect(self, rect):
        self.border_rect.setRect(rect)

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Choose Teto", "", "Teto (*.mp4 *.webm)"
        )
        if file:
            self.path_of_video = file
            self.player.setSource(QUrl.fromLocalFile(file))
            self.player.play()
            self.start_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())