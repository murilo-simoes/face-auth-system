import cv2
import base64
import numpy as np
import os
import tempfile
from typing import List, Tuple, Optional

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


def validate_video_file(file_path: str, max_size_mb: int = 15) -> Tuple[bool, Optional[str]]:
    """
    Valida arquivo de vídeo: tamanho e duração.
    Retorna (is_valid, error_message).
    """
    try:
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False, f"Vídeo excede {max_size_mb} MB"
        
        if file_size == 0:
            return False, "Arquivo vazio"
        
        # Verificar duração usando OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Não foi possível abrir o vídeo"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        if duration < 1.0:
            return False, "Vídeo deve ter pelo menos 1 segundo de duração"
        
        if duration > 10.0:  # Limite superior razoável
            return False, "Vídeo excede duração máxima permitida"
        
        return True, None
    
    except Exception as e:
        return False, f"Erro ao validar vídeo: {str(e)}"


def extract_frames_from_video(video_path: str, num_frames: int = 12) -> Tuple[List[np.ndarray], float]:
    """
    Extrai frames uniformemente distribuídos do vídeo.
    Retorna (frames_list, fps).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return [], 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return [], fps
    
    # Calcular índices dos frames a extrair (distribuição uniforme)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames, fps


def save_temp_video(file_storage, video_format: str = "mp4") -> Optional[str]:
    """
    Salva arquivo de vídeo temporário com nome seguro.
    Retorna caminho do arquivo ou None em caso de erro.
    """
    try:
        # Criar arquivo temporário com extensão apropriada
        suffix = f".{video_format}"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb')
        
        # Salvar conteúdo do upload
        file_storage.seek(0)
        temp_file.write(file_storage.read())
        temp_file.close()
        
        # Restringir permissões (apenas leitura para o dono)
        os.chmod(temp_file.name, 0o600)
        
        return temp_file.name
    
    except Exception as e:
        print(f"Erro ao salvar vídeo temporário: {e}")
        return None


