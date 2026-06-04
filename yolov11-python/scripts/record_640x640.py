import cv2
import os
import time
import sys

def get_usb_camera_index():
    """
    Scans camera indices to find active video sources.
    If multiple cameras are detected, prioritizes index 1 (usually the external USB camera on PC).
    """
    available = []
    # Test indices 0 to 4
    for i in range(5):
        # On Windows, cv2.CAP_DSHOW can speed up opening the camera
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            available.append(i)
            cap.release()
            
    if len(available) > 1:
        # If there's an integrated webcam (0) and a USB camera (1), choose the USB camera (1)
        selected = available[1]
        print(f"[i] Cámaras detectadas: {available}. Seleccionando cámara USB en índice {selected}.")
        return selected
    elif len(available) == 1:
        selected = available[0]
        print(f"[i] Única cámara detectada: índice {selected}.")
        return selected
    else:
        print("[!] No se detectaron cámaras activas. Probando índice 1 por defecto para cámara USB.")
        return 1

def record_video(output_dir, target_fps=30.0):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"[+] Carpeta creada: {output_dir}")
        except Exception as e:
            print(f"[❌] Error al crear la carpeta {output_dir}: {e}")
            sys.exit(1)

    # Detect camera index
    camera_index = get_usb_camera_index()
    
    print(f"[i] Abriendo cámara en índice {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        # Fallback to index 0 if index 1 failed
        if camera_index != 0:
            print("[!] Error al abrir cámara USB. Intentando cámara por defecto (índice 0)...")
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            print(f"[❌] Error: No se pudo abrir ninguna cámara.")
            sys.exit(1)

    # Allow camera to warm up
    time.sleep(1.0)

    # Generate output file name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"captura_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Configure Video Writer (Platform-dependent for optimal H.264 compatibility)
    if os.name == 'nt':
        # On Windows, use Microsoft Media Foundation (MSMF) to encode standard H.264 (avc1) natively
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, cv2.CAP_MSMF, fourcc, target_fps, (640, 640))
    else:
        # On Linux/Raspberry Pi/macOS, use default FFMPEG backend with mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (640, 640))

    if not out.isOpened():
        print("[❌] Error: No se pudo iniciar el codificador de video (VideoWriter).")
        cap.release()
        sys.exit(1)

    window_name = "Grabando Video 640x640 (Q para salir)"
    
    print("=" * 60)
    print(f"  INICIANDO GRABACIÓN DE VIDEO 640x640  ")
    print("=" * 60)
    print(f"[✓] Grabando en: {output_path}")
    print("[i] Presiona 'q' / 'Esc' o cierra la ventana de la cámara para detener la grabación.")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()

    # Create named window first so window properties can be checked
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[❌] Error al leer el frame de la cámara.")
                break

            # Get dimensions
            h, w, _ = frame.shape

            # Crop central square to maintain aspect ratio without distortion
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # Resize to exactly 640x640
            resized_frame = cv2.resize(cropped_frame, (640, 640))

            # Write to output file
            out.write(resized_frame)
            frame_count += 1

            # Prepare visualization frame (draw recording indicator dot)
            display_frame = resized_frame.copy()
            dot_color = (0, 0, 255) # Red
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(display_frame, (30, 30), 8, dot_color, -1)
            else:
                cv2.circle(display_frame, (30, 30), 8, dot_color, 2)
            
            cv2.putText(display_frame, "REC", (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)
            
            # Show live preview
            cv2.imshow(window_name, display_frame)

            # Process window events and check if 'q' or 'Esc' (ASCII 27) is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n[i] Deteniendo grabación por petición del usuario...")
                break

            # Check if window was closed via the "X" button (WND_PROP_VISIBLE < 1)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[i] Ventana cerrada por el usuario. Deteniendo grabación...")
                    break
            except Exception:
                print("\n[i] Ventana cerrada por el usuario. Deteniendo grabación...")
                break

            # Check if a file named 'stop.txt' exists to stop cleanly
            if os.path.exists(os.path.join(output_dir, "stop.txt")):
                print("\n[i] Archivo stop.txt detectado. Deteniendo grabación...")
                try:
                    os.remove(os.path.join(output_dir, "stop.txt"))
                except Exception:
                    pass
                break

    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        duration = time.time() - start_time
        actual_fps = frame_count / duration if duration > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"[✓] Video guardado correctamente en: {output_path}")
        print(f"[i] Duración: {duration:.2f} segundos")
        print(f"[i] Total de frames: {frame_count}")
        print(f"[i] FPS promedio: {actual_fps:.2f}")
        print("=" * 60)

if __name__ == "__main__":
    # Default output directory
    output_directory = r"C:\Users\angel\OneDrive\Desktop\tesis\captura_videos"
    record_video(output_directory)
