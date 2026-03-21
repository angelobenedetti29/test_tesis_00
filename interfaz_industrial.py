import sys
import os
import cv2
import numpy as np
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                               QVBoxLayout, QPushButton, QLabel, QFrame, QProgressBar, 
                               QSpacerItem, QSizePolicy, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

QSS = """
QMainWindow {
    background-color: #D8BFD8;
}

#Sidebar {
    background-color: #1A0B2E;
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
}

#Sidebar QLabel {
    color: #E0B0FF;
    font-family: 'Segoe UI', Roboto, sans-serif;
    font-weight: bold;
}

#Sidebar QPushButton {
    background-color: transparent;
    color: #E0B0FF;
    font-family: 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    font-weight: bold;
    text-align: left;
    padding: 12px 20px;
    border: 2px solid transparent;
    border-radius: 10px;
    margin: 5px 10px;
}

#Sidebar QPushButton:hover {
    background-color: #2D1B4E;
    border: 2px solid #B026FF;
    color: #FFFFFF;
}

/* Feedback Visual para la Cámara Activa (Checkeable) */
#Sidebar QPushButton:checked {
    background-color: #2D1B4E;
    border: 2px solid #00FFCC; /* Borde más intenso para indicar activo */
    color: #FFFFFF;
}

#BtnLogout {
    background-color: rgba(176, 38, 255, 0.1);
    border: 1px solid #B026FF;
    text-align: center;
}

#BtnLogout:hover {
    background-color: #B026FF;
    color: white;
}

#VideoFrame {
    background-color: #2A2A35;
    border: 2px solid #5A4080;
    border-radius: 15px;
}

#GalleryFrame {
    background-color: #1A0B2E;
    border: 1px solid #5A4080;
    border-radius: 10px;
}

/* ScrollBar Vertical Customization */
QScrollBar:vertical {
    border: none;
    background: #1A0B2E;
    width: 10px;
    border-radius: 5px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #B026FF;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #A45EE5;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* Video Buttons */
.VideoBtn {
    background-color: rgba(26, 11, 46, 0.7);
    color: #E0B0FF;
    font-family: 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    font-weight: 600;
    text-align: left;
    padding: 10px 15px;
    border: 1px solid transparent;
    border-radius: 10px;
    margin: 2px 5px;
}

.VideoBtn:hover {
    background-color: #A45EE5;
    border: 1px solid #D8BFD8;
    color: white;
}

.VideoBtn:pressed {
    background-color: #7B2CBF;
    border: 2px solid #00FFCC;
    color: white;
}

.Card {
    background-color: #1A0B2E;
    border: 1px solid #7B2CBF;
    border-radius: 15px;
}

.Card QLabel {
    color: #E0B0FF;
    font-family: 'Segoe UI', Roboto, sans-serif;
}

QProgressBar {
    background-color: #2D1B4E;
    border-radius: 5px;
    color: transparent;
    height: 10px;
}

QProgressBar::chunk {
    background-color: #00FFCC;
    border-radius: 5px;
}
"""

class YOLODetectionThread(QThread):
    change_pixmap_signal = Signal(QImage)

    def __init__(self, source_file="yolov11-python/data/videos/road.mp4"):
        super().__init__()
        self.source_file = source_file
        self.running = True

    def run(self):
        model_path = "yolov11-python/yolo11n.onnx"
        names_path = "yolov11-python/data/class.names"
        
        IMAGE_SIZE = 640
        NAMES = []
        try:
            with open(names_path, "r") as f:
                NAMES = [cname.strip() for cname in f.readlines()]
        except Exception as e:
            return

        COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
        
        try:
            model = cv2.dnn.readNet(model_path)
        except Exception as e:
            return

        # Adaptación para Cámara (0) o Archivo
        cv_source = 0 if self.source_file == "0" else self.source_file
        cap = cv2.VideoCapture(cv_source)
        
        if not cap.isOpened():
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            image = frame.copy()
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
            model.setInput(blob)
            preds = model.forward()
            preds = preds.transpose((0, 2, 1))

            image_height, image_width, _ = image.shape
            x_factor = image_width / IMAGE_SIZE
            y_factor = image_height / IMAGE_SIZE

            rows = preds[0].shape[0]
            class_ids, confs, boxes = list(), list(), list()

            for i in range(rows):
                row = preds[0][i]
                conf = row[4]
                classes_score = row[4:]
                _, _, _, max_idx = cv2.minMaxLoc(classes_score)
                class_id = max_idx[1]
                if classes_score[class_id] > 0.25:
                    confs.append(classes_score[class_id])
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    boxes.append([left, top, width, height])

            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    box = boxes[i]
                    class_id = class_ids[i]
                    score = confs[i]
                    left, top, width, height = box[0], box[1], box[2], box[3]
                    
                    cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], 2)
                    name = f"{NAMES[class_id]} {round(float(score), 3)}"
                    cv2.putText(image, name, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[class_id], 2)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.change_pixmap_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class FactoryControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SISTEMA DE CONTROL INDUSTRIAL - PYSIDE6")
        self.resize(1200, 800)
        self.setStyleSheet(QSS)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 20, 20)
        main_layout.setSpacing(20)

        # 1. BARRA LATERAL IZQUIERDA (SIDEBAR)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 30, 15, 30)

        logo_label = QLabel("FACTORY CONTROL\nPYSIDE6")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        sidebar_layout.addWidget(logo_label)

        self.nav_buttons = []
        nav_titles = ["📸 Cámara en Vivo", "📊 Estadísticas", "💻 Datos del PC", "👥 Usuarios"]
        for i, btn_text in enumerate(nav_titles):
            btn = QPushButton(btn_text)
            # Solo la cámara la hacemos interactiva como switch (Checkable)
            if i == 0:
                btn.setCheckable(True)
                btn.clicked.connect(self.toggle_camera)
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        user_label = QLabel("USUARIO: ADMIN")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("font-size: 14px; margin-bottom: 10px; color: white;")
        sidebar_layout.addWidget(user_label)

        logout_btn = QPushButton("CERRAR SESIÓN")
        logout_btn.setObjectName("BtnLogout")
        sidebar_layout.addWidget(logout_btn)

        main_layout.addWidget(sidebar)

        # 2. ÁREA CENTRAL Y PANEL DERECHO
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 20, 0, 0)
        content_layout.setSpacing(20)

        # ÁREA DE VIDEO CENTRAL + LISTA DE VIDEOS
        video_area_layout = QVBoxLayout()
        video_area_layout.setSpacing(15)

        video_frame = QFrame()
        video_frame.setObjectName("VideoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(15, 15, 15, 15)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1E1E28; border-radius: 10px;")
        video_layout.addWidget(self.video_label)
        
        video_area_layout.addWidget(video_frame, stretch=7)

        # GALERÍA DE VIDEOS (QScrollArea)
        gallery_frame = QFrame()
        gallery_frame.setObjectName("GalleryFrame")
        gallery_layout = QVBoxLayout(gallery_frame)
        gallery_layout.setContentsMargins(15, 10, 15, 10)
        
        gallery_title = QLabel("GALERÍA DE VIDEOS")
        gallery_title.setStyleSheet("color: #E0B0FF; font-weight: bold; font-size: 14px;")
        gallery_layout.addWidget(gallery_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: transparent; border: none;")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        self.videos_layout = QVBoxLayout(scroll_content)
        self.videos_layout.setAlignment(Qt.AlignTop)
        
        # Escaneo de Carpeta usando os
        videos_path = "yolov11-python/data/videos"
        if os.path.exists(videos_path):
            videos = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
            for video in videos:
                btn = QPushButton(video)
                btn.setProperty("class", "VideoBtn")
                btn.clicked.connect(lambda checked=False, btn_ref=btn: self.play_internal_target(btn_ref.text()))
                self.videos_layout.addWidget(btn)
        else:
            no_videos_lbl = QLabel("No se encontró la ruta: data/videos")
            no_videos_lbl.setStyleSheet("color: gray;")
            self.videos_layout.addWidget(no_videos_lbl)

        scroll_area.setWidget(scroll_content)
        gallery_layout.addWidget(scroll_area)

        video_area_layout.addWidget(gallery_frame, stretch=3)

        content_layout.addLayout(video_area_layout, stretch=7)

        # PANEL DERECHO DE INFORMACIÓN
        info_panel_layout = QVBoxLayout()
        info_panel_layout.setSpacing(15)

        card_chart = QFrame()
        card_chart.setProperty("class", "Card")
        chart_layout = QVBoxLayout(card_chart)
        chart_title = QLabel("MINI CHART")
        chart_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        # Fix SyntaxWarning
        chart_placeholder = QLabel("--- \\/\\/\\/ ---")
        chart_placeholder.setAlignment(Qt.AlignCenter)
        chart_placeholder.setStyleSheet("color: #B026FF; font-size: 24px;")
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(chart_placeholder)
        info_panel_layout.addWidget(card_chart, stretch=1)

        card_server = QFrame()
        card_server.setProperty("class", "Card")
        server_layout = QVBoxLayout(card_server)
        server_title = QLabel("SERVIDOR STATUS")
        server_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        server_layout.addWidget(server_title)
        
        for stat, value in [("Server Status:", 75), ("Gestión Eventos:", 45), ("Alertas:", 15)]:
            stat_layout = QHBoxLayout()
            lbl = QLabel(stat)
            lbl.setStyleSheet("font-size: 11px;")
            stat_layout.addWidget(lbl)
            
            bar = QProgressBar()
            bar.setValue(value)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            stat_layout.addWidget(bar)
            server_layout.addLayout(stat_layout)
            
        info_panel_layout.addWidget(card_server, stretch=1)

        card_users = QFrame()
        card_users.setProperty("class", "Card")
        users_layout = QVBoxLayout(card_users)
        users_title = QLabel("USERS DEFINIDOS")
        users_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        users_layout.addWidget(users_title)
        
        counters_layout = QHBoxLayout()
        c1_layout = QVBoxLayout()
        c1_layout.addWidget(QLabel("Monto:"))
        c1_lbl = QLabel("1034")
        c1_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        c1_layout.addWidget(c1_lbl)
        counters_layout.addLayout(c1_layout)
        
        c2_layout = QVBoxLayout()
        c2_layout.addWidget(QLabel("Habial:"))
        c2_lbl = QLabel("80")
        c2_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        c2_layout.addWidget(c2_lbl)
        counters_layout.addLayout(c2_layout)

        users_layout.addLayout(counters_layout)
        info_panel_layout.addWidget(card_users, stretch=1)

        content_layout.addLayout(info_panel_layout, stretch=3)
        main_layout.addLayout(content_layout)

        self.yolo_thread = None
        self.play_internal_target("road.mp4")

    # Lógica Botón Cámara (Manejo de estado)
    def toggle_camera(self, checked):
        if checked:
            # Si activamos cámara (source 0)
            self.play_internal_target("0")
        else:
            # Si apagamos cámara manualmente, detener el hilo y limpiar recuadro
            if self.yolo_thread is not None and self.yolo_thread.isRunning():
                self.yolo_thread.stop()
            self.video_label.clear()
            self.video_label.setText("Cámara Apagada")
            self.video_label.setStyleSheet("background-color: #1E1E28; border-radius: 10px; color: #E0B0FF;")

    def play_internal_target(self, video_name):
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
            
        # Si se hace click a un video de galería (no "0"), desmarcar el botón de la cámara
        if video_name != "0":
            self.nav_buttons[0].setChecked(False)
            source_path = f"yolov11-python/data/videos/{video_name}"
        else:
            source_path = "0"
            
        self.yolo_thread = YOLODetectionThread(source_path)
        self.yolo_thread.change_pixmap_signal.connect(self.update_image)
        self.yolo_thread.start()

    @Slot(QImage)
    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FactoryControlApp()
    window.show()
    sys.exit(app.exec())
