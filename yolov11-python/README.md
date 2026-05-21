# Guía de Uso: Inferencia Acelerada por Hardware (NPU Hailo-8L) en Raspberry Pi 5

Este repositorio contiene la integración de la NPU **Hailo-8L** para la inferencia de modelos YOLO de visión por computadora en tiempo real sobre la Raspberry Pi 5. La arquitectura implementada admite ejecución híbrida (NPU en hardware y CPU tradicional con OpenCV) con optimizaciones concurrentes avanzadas a nivel de hilos para maximizar los FPS y minimizar la carga del procesador.

---

## 📂 Estructura Principal del Workspace

* **`interfaz_industrial.py`**: Interfaz gráfica industrial en PySide6 (Qt) optimizada para control de fábrica. Soporta reproducción fluida de cámaras en vivo y galería de videos con análisis inteligente en segundo plano.
* **`yolov11-python/detection.py`**: Script de consola principal adaptado para correr inferencias rápidas en imágenes, videos o webcams usando archivos de modelo `.hef` (NPU Hailo) o `.onnx` (CPU tradicional).
* **`yolov11-python/data/videos/`**: Contiene los videos de pruebas (incluyendo el archivo optimizado `road640x480.mp4`).

---

## 🖥️ 1. Cómo Ejecutar la Interfaz Gráfica (PySide6)

La interfaz gráfica cuenta con un diseño optimizado que corre el análisis por IA en un hilo secundario (`QThread`), limitador dinámico de FPS y re-escalado eficiente fuera del hilo principal.

### Ejecución Estándar (Modo Acelerado NPU):
```bash
python3 interfaz_industrial.py
```

### Ejecución con Telemetría Activa (Recomendado para Tesis):
Para permitir que las herramientas de telemetría de Hailo capturen el rendimiento en tiempo real mientras usas la interfaz gráfica:
```bash
HAILO_MONITOR=1 python3 interfaz_industrial.py
```
*Haz clic en **`road640x480.mp4`** en la galería de la derecha para ver la máxima fluidez con el menor consumo de CPU.*

---

## ⚙️ 2. Cómo Ejecutar Pruebas por Consola (`detection.py`)

El script principal de consola permite realizar diagnósticos veloces y comparativas de rendimiento directo.

### A. Ejecución en NPU (Aceleración por Hardware Hailo-8L)
Utiliza el modelo precompilado oficial `.hef` de YOLOv8s optimizado para la arquitectura Hailo-8L:
```bash
python3 yolov11-python/detection.py --model /usr/share/hailo-models/yolov8s_h8l.hef --source yolov11-python/data/videos/road640x480.mp4
```

### B. Ejecución en CPU (Retrocompatibilidad ONNX con OpenCV)
Utiliza el modelo tradicional de YOLOv11 en formato de red `.onnx`:
```bash
python3 yolov11-python/detection.py --model yolov11-python/yolo11n.onnx --source yolov11-python/data/videos/road640x480.mp4
```

---

## 📊 3. Cómo Monitorear el Chip de IA (NPU) en Tiempo Real

Para medir la carga de procesamiento, temperatura y consumo eléctrico de la NPU Hailo-8L (ideal para justificaciones técnicas y demostraciones en tu defensa de tesis):

1. **Asegúrate de haber lanzado la aplicación con la variable habilitadora:**
   ```bash
   HAILO_MONITOR=1 python3 interfaz_industrial.py
   ```
2. **Abre otra terminal paralela y ejecuta el comando de telemetría oficial:**
   ```bash
   hailortcli monitor
   ```
   *Esto desplegará una pantalla interactiva con:*
   * **Utilization (%)**: Carga matemática de procesamiento del chip de IA.
   * **Power (W)**: Consumo eléctrico dinámico del módulo en Watts.
   * **Temp (°C)**: Temperatura de funcionamiento del chip.
   * **FPS**: Tasa de fotogramas por segundo lograda.

---

## ⚡ 4. Recomendaciones de Optimización Industrial

* **Uso de Formatos Nativos (`640x480`):** Al capturar video o usar webcams, prefiere resoluciones nativas de `640x480` píxeles. Dado que la red neuronal YOLO opera internamente en `640x640`, alimentar un video de menor calidad reduce a la décima parte la carga de decodificación en la CPU sin perder un solo punto de precisión en la detección de la IA.
* **Comando para convertir videos propios a formato optimizado:**
  ```bash
  ffmpeg -y -i <tu_video_original.mp4> -vf scale=640:480 -c:v libx264 -crf 23 <video_optimizado_640x480.mp4>
  ```
