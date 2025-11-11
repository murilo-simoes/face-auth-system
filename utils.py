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
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[DEBUG] Validação de vídeo - Tamanho: {file_size_mb:.2f} MB")
        
        if file_size > max_size_bytes:
            print(f"[ERROR] Vídeo muito grande: {file_size_mb:.2f} MB > {max_size_mb} MB")
            return False, f"Vídeo excede {max_size_mb} MB"
        
        if file_size == 0:
            print("[ERROR] Arquivo de vídeo está vazio")
            return False, "Arquivo vazio"
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("[ERROR] Não foi possível abrir o vídeo com OpenCV")
            return False, "Não foi possível abrir o vídeo"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"[DEBUG] Vídeo - FPS: {fps}, Frames: {frame_count}, Duração: {duration:.2f}s")
        
        cap.release()
        
        if duration < 1.0:
            print(f"[ERROR] Vídeo muito curto: {duration:.2f}s < 1.0s")
            return False, "Vídeo deve ter pelo menos 1 segundo de duração"
        
        if duration > 10.0:
            print(f"[ERROR] Vídeo muito longo: {duration:.2f}s > 10.0s")
            return False, "Vídeo excede duração máxima permitida"
        
        print("[DEBUG] Validação de vídeo: OK")
        return True, None
    
    except Exception as e:
        print(f"[ERROR] Exceção ao validar vídeo: {str(e)}")
        return False, f"Erro ao validar vídeo: {str(e)}"


def extract_frames_from_video(video_path: str, num_frames: int = 12) -> Tuple[List[np.ndarray], float]:
    """
    Extrai frames uniformemente distribuídos do vídeo.
    Retorna (frames_list, fps).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] Não foi possível abrir vídeo para extrair frames")
        return [], 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Corrigir FPS inválido (WebM às vezes reporta FPS errado)
    if fps <= 0 or fps > 120:
        # Tentar calcular FPS real lendo alguns frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, _ = cap.read()
        if ret and total_frames > 0:
            # Ler alguns frames para estimar FPS
            test_frames = min(30, total_frames)
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            for _ in range(test_frames):
                ret, _ = cap.read()
                if not ret:
                    break
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end_time > start_time:
                estimated_fps = test_frames / (end_time - start_time)
                if 1 <= estimated_fps <= 120:
                    fps = estimated_fps
                    print(f"[DEBUG] FPS corrigido: {fps:.2f} (estimado)")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Se ainda inválido, usar FPS padrão
        if fps <= 0 or fps > 120:
            fps = 30.0
            print(f"[DEBUG] FPS inválido, usando padrão: {fps}")
    
    print(f"[DEBUG] Extraindo frames - Total: {total_frames}, FPS: {fps:.2f}, Solicitados: {num_frames}")
    
    if total_frames == 0:
        print("[ERROR] Vídeo não tem frames")
        cap.release()
        return [], fps
    
    # Extrair frames de forma mais robusta
    # Tentar usar índices calculados primeiro
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"[DEBUG] Tentando extrair frames nos índices: {frame_indices.tolist()}")
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
        else:
            # Se falhar, tentar ler frame sequencialmente
            print(f"[WARNING] Falha ao ler frame {idx}, tentando método alternativo")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(idx + 1):
                ret, frame = cap.read()
                if not ret:
                    break
            if ret and frame is not None:
                frames.append(frame)
    
    # Se ainda não conseguiu frames suficientes, ler sequencialmente
    if len(frames) < num_frames:
        print(f"[WARNING] Apenas {len(frames)} frames extraídos, tentando leitura sequencial")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        step = max(1, total_frames // num_frames)
        for i in range(0, total_frames, step):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                if len(frames) >= num_frames:
                    break
    
    print(f"[DEBUG] Frames extraídos com sucesso: {len(frames)}/{num_frames}")
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


