# Proyecto de Tesis: Detección de Tostadas con YOLOv11 & IoT

Este proyecto está diseñado para implementarse en una Raspberry Pi y utiliza una **Arquitectura Limpia (Clean Architecture)** en Python para separar la lógica de dominio de las librerías de infraestructura (como OpenCV, YOLO y FastAPI) y de la interfaz gráfica local.

---

## Estructura del Proyecto

```
yolov11-python/
│
├── backend/                         # Lógica de Negocio y Servidor
│   ├── app/                         # Capa API (FastAPI)
│   │   ├── main.py                  # Punto de entrada de la API backend
│   │   └── api/                     # Rutas y controladores de la API
│   │
│   ├── domain/                      # Capa de Dominio (Modelos e interfaces abstractas)
│   │   ├── entities/                # Entidades puras (device.py, detection.py)
│   │   └── interfaces/              # Abstracciones de componentes (IIoTController, etc.)
│   │
│   ├── use_cases/                   # Reglas de aplicación (detect_and_notify.py, etc.)
│   │
│   └── infrastructure/              # Implementaciones concretas de la tecnología
│       ├── ai/                      # Detector YOLO usando OpenCV DNN (.onnx)
│       ├── iot/                     # Adaptadores de comunicación IoT (Mock/Físico)
│       └── http/                    # Adaptadores de comunicación HTTP saliente
│
├── frontend/                        # Interfaz gráfica local para la Raspberry Pi
│   ├── main.py                      # Punto de entrada de la GUI
│   ├── gui_app.py                   # Lógica y diseño visual (CustomTkinter/Tkinter)
│   └── assets/                      # Iconos y recursos estáticos
│
├── ai_training/                     # Entrenamiento de Modelos (Proceso aislado)
│   ├── datasets/                    # Datasets de entrenamiento (data.yaml)
│   ├── scripts/                     # Scripts para entrenar, validar y exportar modelos
│   └── models/                      # Pesos base (.pt) y modelos exportados (.onnx)
│
├── scripts/                         # Diagnósticos y ejecutables sueltos para pruebas rápidas
│   ├── detection_onnx.py            # Detección rápida e independiente con ONNX
│   └── detection_pt.py              # Detección rápida e independiente con PyTorch
│
├── multimedia/                      # Archivos multimedia locales
│   ├── input/                       # Videos e imágenes de prueba de entrada
│   ├── output/                      # Videos/imágenes procesados con bounding boxes
│   ├── videos/                      # Videos de prueba generales
│   └── images/                      # Imágenes de prueba generales
│
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación general
```

---

## Ejecución del Proyecto

### 1. Iniciar el Servidor Backend (API)
El backend corre un servidor FastAPI que expone endpoints para recibir alertas e imágenes del exterior, y permite controlar los relés de IoT mediante HTTP.
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Una vez iniciado, puedes consultar la documentación interactiva en: http://localhost:8000/docs

### 2. Iniciar la Interfaz Gráfica Local (Frontend)
El frontend es una aplicación de escritorio basada en **CustomTkinter** que se ejecuta localmente en la pantalla de la Raspberry Pi.

* Para ejecutar usando la **Webcam/Cámara de la Pi** (ID `0`):
  ```bash
  python -m frontend.main --source 0
  ```
* Para ejecutar pasando como entrada un **video de prueba**:
  ```bash
  python -m frontend.main --source multimedia/videos/road.mp4
  ```

### 3. Ejecutar Pruebas Rápidas de Diagnóstico
Si deseas verificar que la cámara y el modelo funcionan en tiempo real sin levantar toda la arquitectura backend/frontend:
* Prueba rápida con **ONNX**:
  ```bash
  python scripts/detection_onnx.py --model ai_training/models/tostadas.onnx --names ai_training/models/tostadas.names --source 0
  ```
* Prueba rápida con **PyTorch** (`best.pt`):
  ```bash
  python scripts/detection_pt.py --model ai_training/runs/detect/train/weights/best.pt --source 0
  ```

---

## Entrenamiento y Preparación de Modelos
Si necesitas re-entrenar el detector de YOLOv11:
1. Coloca tu dataset en `ai_training/datasets/` y asegúrate de configurar correctamente el archivo `data.yaml`.
2. Inicia el entrenamiento con:
   ```bash
   python ai_training/scripts/train.py
   ```
   *Nota: El script entrenará el modelo, exportará el resultado a formato ONNX (`tostadas.onnx`) y creará el archivo de etiquetas en `ai_training/models/tostadas.names` automáticamente.*
