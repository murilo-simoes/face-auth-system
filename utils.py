import cv2
import base64
import numpy as np

def decode_base64_image(image_base64: str):
    """Converte string base64 em frame RGB (OpenCV)."""
    try:
        img_data = base64.b64decode(image_base64)
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception as e:
        print(f"Erro ao decodificar imagem: {e}")
        return None

