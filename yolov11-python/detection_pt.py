import cv2
import time
import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Definir argumentos de entrada por consola
    parser = argparse.ArgumentParser(description="Detector de Tostadas en Tiempo Real (PyTorch Nativo)")
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Ruta al video, imagen o '0' para webcam")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Ruta a los pesos best.pt")
    parser.add_argument("--tresh", type=float, default=0.25, help="Umbral de confianza")
    args = parser.parse_args()

    print("=" * 60)
    print("  INICIANDO DETECTOR DE TOSTADAS - PYTORCH NATIVO  ")
    print("=" * 60)
    print(f"[i] Cargando pesos entrenados desde: {args.model}")

    # 2. Cargar el modelo entrenado (.pt)
    try:
        model = YOLO(args.model)
        print("[✔] Modelo cargado con éxito en PyTorch.")
    except Exception as e:
        print(f"[❌] Error al cargar el modelo: {e}")
        exit(1)

    # 3. Detectar la fuente de video/imagen
    source = args.source
    if source == "0":
        source = 0  # Usar webcam integrada
        print("[i] Iniciando captura desde la Webcam (0)...")
    else:
        print(f"[i] Abriendo fuente de entrada: {source}")

    # 4. Iniciar captura y procesamiento
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[❌] Error: No se pudo abrir la fuente {source}")
        exit(1)

    # Determinar si la fuente es una imagen estática
    is_image = False
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp']
    if isinstance(source, str) and source.split('.')[-1].lower() in img_formats:
        is_image = True

    print("\n" + "=" * 60)
    print("  EJECUTANDO DETECTOR (Presiona 'q' para salir)  ")
    print("=" * 60)

    # Ejecutar inferencia en modo generador (stream=True) para mayor rendimiento
    results = model(source, conf=args.tresh, stream=True)

    for r in results:
        # r.plot() dibuja automáticamente las cajas, etiquetas y confianzas de forma optimizada
        annotated_frame = r.plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            boxes=True
        )

        # Mostrar el frame procesado
        cv2.imshow("Detector de Tostadas (YOLOv11)", annotated_frame)

        # Si es una imagen estática, esperar indefinidamente hasta presionar una tecla
        if is_image:
            cv2.waitKey(0)
            break

        # Si es un video, esperar 1ms y salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✔] Ejecución finalizada correctamente.")
