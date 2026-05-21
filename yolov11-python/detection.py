import cv2
import time
import random
import argparse
import numpy as np
# import onnxruntime as ort

# Intentar importar la librería oficial de Hailo de forma segura
try:
    from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams, HailoStreamInterface
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# Clase helper para encapsular la inferencia en la NPU Hailo
class HailoInference:
    def __init__(self, hef_path):
        if not HAILO_AVAILABLE:
            raise RuntimeError("La librería 'hailo_platform' no está instalada.")
        
        self.hef_path = hef_path
        self.hef = HEF(self.hef_path)
        self.vdevice = VDevice().__enter__()
        
        try:
            # Configurar el dispositivo PCIe con la red HEF
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
            
            # Obtener nombres de streams virtuales de entrada y salida
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
            
            # Configurar parámetros (UINT8 para entrada y FLOAT32 para salida)
            self.input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
            self.output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            
            # Activar el grupo de red
            self.activation_ctx = self.network_group.activate()
            self.activation_ctx.__enter__()
            
            # Inicializar la tubería de inferencia
            self.infer_ctx = InferVStreams(self.network_group, self.input_params, self.output_params)
            self.infer_pipeline = self.infer_ctx.__enter__()
        except Exception as e:
            self.release()
            raise e

    def infer(self, img_rgb_640):
        # La NPU espera una forma (1, 640, 640, 3)
        input_data = np.expand_dims(img_rgb_640, axis=0)
        outputs = self.infer_pipeline.infer({self.input_vstream_info.name: input_data})
        return outputs[self.output_vstream_info.name]

    def release(self):
        # Liberar los contextos de forma ordenada y segura
        if hasattr(self, 'infer_ctx') and self.infer_ctx:
            try:
                self.infer_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self.infer_ctx = None
        if hasattr(self, 'activation_ctx') and self.activation_ctx:
            try:
                self.activation_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self.activation_ctx = None
        if hasattr(self, 'vdevice') and self.vdevice:
            try:
                self.vdevice.__exit__(None, None, None)
            except Exception:
                pass
            self.vdevice = None

# Function to load the source (image, video, or webcam)
def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # List of image formats
    key = 1 # 1 = Video, 0 = Image
    frame = None
    cap = None

    # If source is a webcam
    if(source_file == "0"):
        image_type = False  # Not an image, it's video from the webcam
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats  # Check if source is an image

    # Open image or video source
    if(image_type):
        frame = cv2.imread(source_file)  # Read image
        key = 0  # Set key for image
    else:
        cap = cv2.VideoCapture(source_file)  # Open video capture for video or webcam

    return image_type, key, frame, cap

if __name__ == '__main__':
    # Add argument parser for command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")  # Video source
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")  # File containing class names
    parser.add_argument("--model", type=str, default="yolo11n.onnx", help="Pretrained Model")  # Model file (.hef o .onnx)
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")  # Confidence threshold for detection
    parser.add_argument("--thickness", type=int, default=2, help="Line Thickness on Bounding Boxes")  # Thickness of bounding box lines
    args = parser.parse_args()    

    IMAGE_SIZE = 640  # Set the image size for the model
    NAMES = []
    # Read the object names from the specified file
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]  # Generate random colors for each object class

    # Detectar dinámicamente si cargamos en Hailo (NPU) o en OpenCV (CPU)
    use_hailo = args.model.endswith('.hef')

    if use_hailo:
        if not HAILO_AVAILABLE:
            raise RuntimeError("Especificaste un modelo .hef, pero la librería 'hailo_platform' no está disponible en este sistema.")
        print(f"[*] Cargando modelo en NPU Hailo-8L: {args.model}")
        hailo_model = HailoInference(args.model)
    else:
        print(f"[*] Cargando modelo en CPU con OpenCV: {args.model}")
        model = cv2.dnn.readNet(args.model)

    source_file = args.source    
    # Load the source (image, video, or webcam)
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True

    while(1):
        # For video input, read the next frame
        if not image_type:
            (grabbed, frame) = cap.read()

        # Exit if no more frames are grabbed
        if not grabbed:
            break

        # Make a copy of the frame
        image = frame.copy()
        image_height, image_width, _ = image.shape
    
        # Initialize lists for detection results
        class_ids, confs, boxes = list(), list(), list()

        if use_hailo:
            # --- FLUJO DE INFERENCIA EN NPU HAILO-8L ---
            # Preprocesamiento: redimensionar a 640x640 y convertir de BGR a RGB
            img_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Ejecutar inferencia en la NPU
            out_tensor = hailo_model.infer(img_rgb)
            
            # El postprocesamiento NMS ya viene compilado dentro del HEF.
            # out_tensor[0] contiene una lista de longitud 80 (clases), 
            # donde cada elemento es un ndarray de forma (N, 5) -> [ymin, xmin, ymax, xmax, confidence]
            detections = out_tensor[0]
            for cid in range(len(detections)):
                class_detections = detections[cid]
                for det in class_detections:
                    ymin, xmin, ymax, xmax, confidence = det
                    if confidence >= args.tresh:
                        # Convertir coordenadas normalizadas a píxeles
                        left = int(xmin * image_width)
                        top = int(ymin * image_height)
                        width = int((xmax - xmin) * image_width)
                        height = int((ymax - ymin) * image_height)
                        
                        boxes.append(np.array([left, top, width, height]))
                        confs.append(float(confidence))
                        class_ids.append(cid)
            
            # En modo Hailo, NMS ya está hecho. Creamos un mapeo plano de índices directos.
            indexes = list(range(len(boxes)))
        else:
            # --- FLUJO DE INFERENCIA EN CPU (ONNX ORIGINAL) ---
            # Prepare image as input for the YOLO model
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)

            # Set input to the model and perform forward pass
            model.setInput(blob)
            preds = model.forward()
            preds = preds.transpose((0, 2, 1))  # Adjust output shape

            # Calculate scaling factors based on image size
            x_factor = image_width / IMAGE_SIZE
            y_factor = image_height / IMAGE_SIZE

            rows = preds[0].shape[0]

            # Iterate over each prediction row
            for i in range(rows):
                row = preds[0][i]
                
                # Extract class scores and find the class with the highest score
                classes_score = row[4:]
                _,_,_, max_idx = cv2.minMaxLoc(classes_score)
                class_id = max_idx[1]
                if (classes_score[class_id] > args.tresh):  # Filter out weak predictions
                    confs.append(classes_score[class_id])  # Store confidence
                    class_ids.append(class_id)  # Store class ID
                    
                    # Extract bounding box coordinates
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)  # Store bounding box coordinates

            # Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confs, args.tresh, 0.5)         

        # Draw bounding boxes and labels on the image
        for i in indexes:
            box = boxes[i]
            class_id = class_ids[i]
            score = confs[i]

            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Draw bounding box on the image
            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            
            # Prepare label with class name and confidence score
            name = NAMES[class_id]    
            score = round(float(score), 3)  # Round confidence score            
            name += f' {str(score)}'

            # Add text label above the bounding box
            font_size = args.thickness / 2.5
            margin = args.thickness * 2
            cv2.putText(image, name, (left, top - margin), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[class_id], args.thickness)

        # Indicate the image has been processed
        if image_type:
            grabbed = False
            
        # Display the processed image with detections
        cv2.imshow("Detected", image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(key) ==  ord('q'):
            break

    # Clean up resources safely
    if use_hailo:
        print("[*] Liberando recursos de la NPU Hailo-8L...")
        hailo_model.release()
        
    cv2.destroyAllWindows()
    print("[*] Proceso finalizado con éxito.")