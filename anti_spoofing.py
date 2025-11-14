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

YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.6"))
CLASS_NAMES = ["fake", "real"]
INVERT_CLASSES = os.getenv("YOLO_INVERT_CLASSES", "false").lower() == "true"
FINAL_SCORE_THRESHOLD = float(os.getenv("FINAL_SCORE_THRESHOLD", "0.6"))

YOLO_MODEL = None
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", None)

if not YOLO_MODEL_PATH:
    possible_paths = [
        "l_version_1_300.pt",
        "models/anti_spoofing_yolo.pt",
        "models/l_version_1_300.pt",
        "anti_spoofing_yolo.pt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            YOLO_MODEL_PATH = path
            print(f"[YOLO] Modelo encontrado automaticamente: {path}")
            break

if YOLO_AVAILABLE and YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
    try:
        YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Modelo carregado: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO] Erro ao carregar modelo: {e}")
        YOLO_MODEL = None
elif YOLO_AVAILABLE and YOLO_MODEL_PATH:
    print(f"[YOLO] Modelo não encontrado em: {YOLO_MODEL_PATH}")


def score_yolo(frames: List[np.ndarray]) -> float:
    if not YOLO_AVAILABLE or YOLO_MODEL is None:
        return 0.5
    
    if len(frames) < 1:
        return 0.5
    
    try:
        scores = []
        detections = []
        sample_frames = frames[::max(1, len(frames)//5)]
        
        for frame in sample_frames:
            results = YOLO_MODEL(frame, stream=True, verbose=False)
            frame_detections = {"fake": [], "real": []}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > YOLO_CONFIDENCE:
                        if INVERT_CLASSES or True:
                            # Inverte as classes somente se solicitado
                            if cls == 0:   # fake
                                frame_detections["real"].append(conf)
                            elif cls == 1: # real
                                frame_detections["fake"].append(conf)
                        else:
                            # Usa as classes corretas do modelo
                            if cls == 0:   # fake
                                frame_detections["fake"].append(conf)
                            elif cls == 1: # real
                                frame_detections["real"].append(conf)
                                    
            max_real = max(frame_detections["real"]) if frame_detections["real"] else 0.0
            max_fake = max(frame_detections["fake"]) if frame_detections["fake"] else 0.0
            
            if max_fake > YOLO_CONFIDENCE:
                scores.append(0.0)
                detections.append(f"F{max_fake:.2f}")
            elif max_real > YOLO_CONFIDENCE:
                scores.append(max_real)
                detections.append(f"R{max_real:.2f}")
            else:
                scores.append(0.5)
                detections.append("N")
        
        if len(scores) == 0:
            return 0.5
        
        avg_score = np.mean(scores)
        no_detections = all(d == "N" for d in detections)
        
        print(f"[YOLO] Detecções: {', '.join(detections)} | Score: {avg_score:.3f}")
        if no_detections:
            print(f"[YOLO] Nenhuma detecção")
        
        result = float(avg_score)
        if not hasattr(score_yolo, '_last_detections'):
            score_yolo._last_detections = {}
        score_yolo._last_detections = {"no_detections": no_detections}
        
        return result
    
    except Exception as e:
        print(f"Erro em YOLO: {e}")
        return 0.5


REAL_THRESHOLD = 0.85  # score mínimo para considerar liveness

def fuse_scores(scores: Dict[str, float], detections_info: Optional[Dict] = None) -> Tuple[float, Optional[str]]:
    yolo_score = scores.get("yolo", 0.5)
    
    # spoof detectado claramente
    if yolo_score < 0.1:
        return 0.0, "spoof_detected_yolo_fake"
    
    # sem detecção suficiente = neutro
    if abs(yolo_score - 0.5) < 0.01:
        return 0.5, "no_detection"

    # real claro
    if yolo_score >= REAL_THRESHOLD:
        return float(yolo_score), "ok"
    
    # suspeito ou baixo score
    return 0.0, "low_confidence"



def process_anti_spoofing(frames: List[np.ndarray], fps: float) -> Dict:
    yolo_score = score_yolo(frames)
    scores = {"yolo": yolo_score}
    detections_info = getattr(score_yolo, '_last_detections', {})
    final_score, reason = fuse_scores(scores, detections_info)
    liveness = final_score >= REAL_THRESHOLD
    
    print(f"[Anti-Spoofing] Score: {scores['yolo']:.3f}, Final: {final_score:.3f}, Liveness: {liveness}, Reason: {reason}")
    
    return {
        "liveness": liveness,
        "final_score": final_score,
        "scores": scores,
        "reason": reason
    }
