import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Warning] YOLO não disponível. Instale com: pip install ultralytics")

# ============================
# CONFIGURAÇÕES
# ============================

YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.6"))

# modelo de classificação: 0 = fake, 1 = real
CLASS_NAMES = ["fake", "real"]

# inverter classes? (0->real, 1->fake)
INVERT_CLASSES = os.getenv("YOLO_INVERT_CLASSES", "true").lower() == "true"

FINAL_SCORE_THRESHOLD = float(os.getenv("FINAL_SCORE_THRESHOLD", "0.6"))

YOLO_MODEL = None
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", None)

# descobrir modelo automaticamente
if not YOLO_MODEL_PATH:
    possible_paths = [
        "best.pt",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            YOLO_MODEL_PATH = path
            print(f"[YOLO] Modelo encontrado automaticamente: {path}")
            break

# carregar modelo
if YOLO_AVAILABLE and YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
    try:
        YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Modelo carregado: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO] Erro ao carregar modelo: {e}")
        YOLO_MODEL = None
elif YOLO_AVAILABLE and YOLO_MODEL_PATH:
    print(f"[YOLO] Modelo não encontrado em: {YOLO_MODEL_PATH}")


# ==========================================
# CLASSIFICAÇÃO VIA YOLOv8-CLS (REAL/FAKE)
# ==========================================

def score_yolo(frames: List[np.ndarray]) -> float:
    """
    Roda o YOLO-CLS em frames.
    Calcula score baseado na média da probabilidade de REAL.
    """
    
    if not YOLO_AVAILABLE or YOLO_MODEL is None:
        return 0.5
    
    if len(frames) < 1:
        return 0.5

    try:
        real_scores = []
        sample_frames = frames[::max(1, len(frames)//5)]  # reduz processamento

        for frame in sample_frames:

            # --- classificação ---
            results = YOLO_MODEL(frame, verbose=False)[0]
            probs = results.probs.data.tolist()

            fake_prob = probs[0]
            real_prob = probs[1]

            # inverter se necessário
            if INVERT_CLASSES:
                fake_prob, real_prob = real_prob, fake_prob

            # manter apenas prob de real
            real_scores.append(real_prob)

        if len(real_scores) == 0:
            return 0.5

        avg_real = float(np.mean(real_scores))

        print(f"[YOLO] Probs Reais: {[round(s,3) for s in real_scores]}")
        print(f"[YOLO] Score final YOLO (média real): {avg_real:.3f}")

        # salvar info para o process_anti_spoofing
        score_yolo._last_detections = {"real_scores": real_scores}

        return avg_real

    except Exception as e:
        print(f"Erro em YOLO: {e}")
        return 0.5


# ===================
# FUSÃO DAS MÉTRICAS
# ===================

REAL_THRESHOLD = 0.6

def fuse_scores(scores: Dict[str, float], detections_info: Optional[Dict] = None) -> Tuple[float, Optional[str]]:
    yolo_score = scores.get("yolo", 0.5)

    # spoof claro
    if yolo_score < 0.1:
        return 0.0, "spoof_detected_yolo_fake"

    # sem confiança suficiente
    if abs(yolo_score - 0.5) < 0.05:
        return 0.5, "no_detection"

    # real claro
    if yolo_score >= REAL_THRESHOLD:
        return float(yolo_score), "ok"

    # suspeito
    return 0.0, "low_confidence"


# =============================================
# PROCESSO COMPLETO DE ANTI-SPOOFING FINAL
# =============================================

def process_anti_spoofing(frames: List[np.ndarray], fps: float) -> Dict:
    """
    frames = lista de frames (numpy arrays)
    fps = fps do vídeo
    """
    
    yolo_score = score_yolo(frames)

    scores = {
        "yolo": yolo_score
    }

    detections_info = getattr(score_yolo, "_last_detections", {})
    final_score, reason = fuse_scores(scores, detections_info)

    liveness = final_score >= REAL_THRESHOLD

    print(f"[Anti-Spoofing] YOLO Score: {yolo_score:.3f} | Final: {final_score:.3f} | Liveness: {liveness} | Reason: {reason}")

    return {
        "liveness": liveness,
        "final_score": final_score,
        "scores": scores,
        "reason": reason
    }
