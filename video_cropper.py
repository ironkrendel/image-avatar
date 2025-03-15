import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsItemGroup, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QUrl, QRectF, QPointF
from PyQt6.QtGui import QBrush, QColor, QPen


class DraggableRectItem(QGraphicsRectItem):
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setBrush(QBrush(QColor(255, 0, 0, 100)))
        self.offset = QPointF()

    def mousePressEvent(self, event):
        # self.offset = event.pos() - self.rect().center()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        new_center = event.scenePos() - self.offset

        scene_rect = self.scene().sceneRect()

        rect_width = self.rect().width()
        rect_height = self.rect().height()

        scale_factor = self.scene().views()[0].transform().m11()
        scaled_scene_rect = QRectF(
            scene_rect.left() * scale_factor,
            scene_rect.top() * scale_factor,
            scene_rect.width() * scale_factor,
            scene_rect.height() * scale_factor
        )

        new_center.setX(max(scaled_scene_rect.left() + rect_width / 2, min(new_center.x(), scaled_scene_rect.right() - rect_width / 2)))
        new_center.setY(max(scaled_scene_rect.top() + rect_height / 2, min(new_center.y(), scaled_scene_rect.bottom() - rect_height / 2)))

        self.setPos(new_center - QPointF(rect_width / 2, rect_height / 2))


class CropRectItemGroup(QGraphicsItemGroup):
    def __init__(self, VideoItemOffset, parent = None):
        super().__init__(parent)
        self.rect = DraggableRectItem(QRectF(0, 0, 200, 150))
        self.rect.setPos(QPointF(VideoItemOffset.x(), VideoItemOffset.y()))
        self.topLeftCircle = QGraphicsEllipseItem()
        self.topLeftCircle.setBrush(QBrush(QColor(0, 255, 0, 100)))
        self.topLeftCircle.setRect(QRectF(0, 0, 15, 15))
        self.topLeftCircle.setPos(QPointF(VideoItemOffset.x(), VideoItemOffset.y()))
        self.addToGroup(self.rect)
        self.addToGroup(self.topLeftCircle)

    def mousePressEvent(self, event):
        self.rect.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.rect.mouseMoveEvent(event)

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

        layout = QVBoxLayout()
        layout.addWidget(self.view)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.select_button)
        layout.addWidget(right_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.video_item.setScale(2.5)
            video_size = self.video_item.size()
            self.scene.setSceneRect(0,75, video_size.width()*2.5, 450)
            self.video_item.setPos(0, 0)

            # self.rect_item = DraggableRectItem(QRectF(0, 0, 200, 150))
            self.rect_item = CropRectItemGroup(self.video_item.scene().sceneRect())
            self.scene.addItem(self.rect_item)

            print(video_size.width()*2.5, video_size.height()*2.5)

    def update_border_rect(self, rect):
        self.border_rect.setRect(rect)

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Choose Video", "", "Teto (*.mp4 *.webm *.mkv)"
        )
        if file:
            self.player.setSource(QUrl.fromLocalFile(file))
            self.player.play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())