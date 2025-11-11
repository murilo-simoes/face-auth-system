"""
Script automatizado para treinar modelo YOLO anti-spoofing.
Automatiza coleta, divisão e treinamento.
"""
import os
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import shutil
import random
from itertools import islice
from ultralytics import YOLO
from time import time
import argparse

# Configurações
OUTPUT_FOLDER = 'Dataset/DataCollect'
SPLIT_FOLDER = 'Dataset/SplitData'
MODEL_OUTPUT = 'models/anti_spoofing_yolo.pt'
CLASSES = ["fake", "real"]
SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}
CONFIDENCE = 0.8
BLUR_THRESHOLD = 35
OFFSET_W = 10
OFFSET_H = 20
FLOATING_POINT = 6

def collect_from_camera(class_id, class_name, min_images=100):
    """Coleta dados da câmera para uma classe específica."""
    print(f"\n{'='*60}")
    print(f"COLETANDO DADOS: {class_name.upper()} (Classe {class_id})")
    print(f"{'='*60}")
    print(f"Pressione 'q' para finalizar, 's' para salvar manualmente")
    print(f"Meta: coletar pelo menos {min_images} imagens")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    detector = FaceDetector()
    count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgOut = img.copy()
        img, bboxs = detector.findFaces(img, draw=False)
        
        listBlur = []
        listInfo = []
        
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]
                
                if score > CONFIDENCE:
                    # Offset
                    offsetW = (OFFSET_W / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)
                    
                    offsetH = (OFFSET_H / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)
                    
                    # Clamp
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)
                    
                    # Blur check
                    imgFace = img[y:y + h, x:x + w]
                    if imgFace.size > 0:
                        blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                        
                        if blurValue > BLUR_THRESHOLD:
                            listBlur.append(True)
                        else:
                            listBlur.append(False)
                        
                        # Normalize
                        ih, iw, _ = img.shape
                        xc, yc = x + w / 2, y + h / 2
                        xcn = min(1.0, round(xc / iw, FLOATING_POINT))
                        ycn = min(1.0, round(yc / ih, FLOATING_POINT))
                        wn = min(1.0, round(w / iw, FLOATING_POINT))
                        hn = min(1.0, round(h / ih, FLOATING_POINT))
                        
                        listInfo.append(f"{class_id} {xcn} {ycn} {wn} {hn}\n")
                        
                        # Draw
                        cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cvzone.putTextRect(
                            imgOut,
                            f'Score: {int(score * 100)}% Blur: {blurValue} Count: {count}',
                            (x, y - 0),
                            scale=2,
                            thickness=3
                        )
                        
                        # Auto-save if not blurry
                        if all(listBlur) and listBlur:
                            timeNow = str(time()).replace('.', '')
                            imgPath = f"{OUTPUT_FOLDER}/{timeNow}.jpg"
                            labelPath = f"{OUTPUT_FOLDER}/{timeNow}.txt"
                            
                            cv2.imwrite(imgPath, img)
                            with open(labelPath, 'w') as f:
                                for info in listInfo:
                                    f.write(info)
                            
                            count += 1
                            print(f"[{count}] Salvo: {imgPath} (blur={blurValue})")
        
        cv2.putText(imgOut, f"Coletadas: {count}/{min_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Coletando: {class_name.upper()}", imgOut)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if count >= min_images:
                break
            else:
                print(f"[AVISO] Apenas {count} imagens coletadas. Deseja continuar? (s/n)")
                if cv2.waitKey(0) & 0xFF == ord('n'):
                    break
        elif key == ord('s') and listInfo:
            timeNow = str(time()).replace('.', '')
            imgPath = f"{OUTPUT_FOLDER}/{timeNow}.jpg"
            labelPath = f"{OUTPUT_FOLDER}/{timeNow}.txt"
            
            cv2.imwrite(imgPath, img)
            with open(labelPath, 'w') as f:
                for info in listInfo:
                    f.write(info)
            
            count += 1
            print(f"[MANUAL {count}] Salvo: {imgPath}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Coleta de {class_name} finalizada: {count} imagens")
    return count

def collect_from_video(video_path, class_id, output_folder):
    """Extrai frames de um vídeo e cria labels automaticamente."""
    print(f"\nProcessando vídeo: {video_path}")
    print(f"Classe: {CLASSES[class_id]} (ID: {class_id})")
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
        return 0
    
    detector = FaceDetector()
    count = 0
    frame_skip = 5  # Processar 1 frame a cada 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_num % frame_skip != 0:
            continue
        
        img, bboxs = detector.findFaces(frame, draw=False)
        
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]
                
                if score > CONFIDENCE:
                    # Offset
                    offsetW = (OFFSET_W / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)
                    
                    offsetH = (OFFSET_H / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)
                    
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)
                    
                    # Blur check
                    imgFace = frame[y:y + h, x:x + w]
                    if imgFace.size > 0:
                        blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                        
                        if blurValue > BLUR_THRESHOLD:
                            # Normalize
                            ih, iw, _ = frame.shape
                            xc, yc = x + w / 2, y + h / 2
                            xcn = min(1.0, round(xc / iw, FLOATING_POINT))
                            ycn = min(1.0, round(yc / ih, FLOATING_POINT))
                            wn = min(1.0, round(w / iw, FLOATING_POINT))
                            hn = min(1.0, round(h / ih, FLOATING_POINT))
                            
                            # Save
                            timeNow = str(time()).replace('.', '')
                            imgPath = f"{output_folder}/{timeNow}.jpg"
                            labelPath = f"{output_folder}/{timeNow}.txt"
                            
                            cv2.imwrite(imgPath, frame)
                            with open(labelPath, 'w') as f:
                                f.write(f"{class_id} {xcn} {ycn} {wn} {hn}\n")
                            
                            count += 1
                            if count % 10 == 0:
                                print(f"  Processados: {count} frames")
                            break
    
    cap.release()
    print(f"[OK] Vídeo processado: {count} imagens extraídas")
    return count

def split_data():
    """Divide dados em train/val/test."""
    print(f"\n{'='*60}")
    print("DIVIDINDO DADOS")
    print(f"{'='*60}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"[ERRO] Pasta de dados não encontrada: {OUTPUT_FOLDER}")
        return False
    
    # Remover pasta de split se existir
    try:
        shutil.rmtree(SPLIT_FOLDER)
    except:
        pass
    
    # Criar diretórios
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{SPLIT_FOLDER}/{split}/images", exist_ok=True)
        os.makedirs(f"{SPLIT_FOLDER}/{split}/labels", exist_ok=True)
    
    # Obter nomes
    listNames = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.jpg')]
    uniqueNames = list(set([name.split('.')[0] for name in listNames]))
    
    if len(uniqueNames) == 0:
        print(f"[ERRO] Nenhuma imagem encontrada em {OUTPUT_FOLDER}")
        return False
    
    # Embaralhar
    random.shuffle(uniqueNames)
    
    # Calcular splits
    lenData = len(uniqueNames)
    lenTrain = int(lenData * SPLIT_RATIO['train'])
    lenVal = int(lenData * SPLIT_RATIO['val'])
    lenTest = int(lenData * SPLIT_RATIO['test'])
    
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining
    
    # Dividir
    lengthToSplit = [lenTrain, lenVal, lenTest]
    Input = iter(uniqueNames)
    Output = [list(islice(Input, elem)) for elem in lengthToSplit]
    
    print(f"Total: {lenData} imagens")
    print(f"Split: Train={len(Output[0])}, Val={len(Output[1])}, Test={len(Output[2])}")
    
    # Copiar arquivos
    sequence = ['train', 'val', 'test']
    for i, out in enumerate(Output):
        for fileName in out:
            srcImg = f'{OUTPUT_FOLDER}/{fileName}.jpg'
            dstImg = f'{SPLIT_FOLDER}/{sequence[i]}/images/{fileName}.jpg'
            srcLabel = f'{OUTPUT_FOLDER}/{fileName}.txt'
            dstLabel = f'{SPLIT_FOLDER}/{sequence[i]}/labels/{fileName}.txt'
            
            if os.path.exists(srcImg):
                shutil.copy(srcImg, dstImg)
            if os.path.exists(srcLabel):
                shutil.copy(srcLabel, dstLabel)
    
    # Criar data.yaml
    dataYaml = f'''path: ../Data
train: ../train/images
val: ../val/images
test: ../test/images

nc: {len(CLASSES)}
names: {CLASSES}'''
    
    yamlPath = f"{SPLIT_FOLDER}/data.yaml"
    with open(yamlPath, 'w') as f:
        f.write(dataYaml)
    
    print(f"[OK] Data.yaml criado: {yamlPath}")
    return True

def train_model(epochs=50):
    """Treina o modelo YOLO."""
    print(f"\n{'='*60}")
    print("TREINANDO MODELO YOLO")
    print(f"{'='*60}")
    
    data_yaml = f"{SPLIT_FOLDER}/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"[ERRO] data.yaml não encontrado: {data_yaml}")
        return False
    
    os.makedirs('models', exist_ok=True)
    
    print(f"Modelo base: yolov8n.pt")
    print(f"Épocas: {epochs}")
    print(f"Data YAML: {data_yaml}")
    print(f"\n[INFO] Isso pode levar algum tempo...")
    
    try:
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='anti_spoofing_yolo',
            project='runs/detect',
            save=True,
            val=True,
        )
        
        # Salvar modelo final
        model.save(MODEL_OUTPUT)
        
        print(f"\n{'='*60}")
        print(f"[SUCESSO] Treinamento concluído!")
        print(f"Modelo salvo em: {MODEL_OUTPUT}")
        print(f"\nConfigure no .env:")
        print(f"YOLO_MODEL_PATH={MODEL_OUTPUT}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Falha no treinamento: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Treinamento automatizado YOLO Anti-Spoofing')
    parser.add_argument('--mode', choices=['camera', 'video', 'skip-collect'], 
                       default='camera', help='Modo de coleta de dados')
    parser.add_argument('--fake-video', help='Caminho para vídeo de spoofing (fake)')
    parser.add_argument('--real-video', help='Caminho para vídeo de rosto real')
    parser.add_argument('--min-images', type=int, default=100, 
                       help='Número mínimo de imagens por classe')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Número de épocas para treinamento')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Pular treinamento (apenas coletar e dividir)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TREINAMENTO AUTOMATIZADO YOLO ANTI-SPOOFING")
    print("="*60)
    
    # Fase 1: Coleta de dados
    if args.mode != 'skip-collect':
        if args.mode == 'camera':
            print("\n[FASE 1] Coleta via câmera")
            print("Você precisará coletar dados para ambas as classes (fake e real)")
            
            # Coletar FAKE
            fake_count = collect_from_camera(0, "fake", args.min_images)
            
            input("\nPressione ENTER para começar a coletar dados REAL...")
            
            # Coletar REAL
            real_count = collect_from_camera(1, "real", args.min_images)
            
            print(f"\n[RESUMO] Fake: {fake_count} imagens, Real: {real_count} imagens")
            
        elif args.mode == 'video':
            if not args.fake_video or not args.real_video:
                print("[ERRO] Modo 'video' requer --fake-video e --real-video")
                return
            
            print("\n[FASE 1] Coleta via vídeos")
            fake_count = collect_from_video(args.fake_video, 0, OUTPUT_FOLDER)
            real_count = collect_from_video(args.real_video, 1, OUTPUT_FOLDER)
            print(f"\n[RESUMO] Fake: {fake_count} imagens, Real: {real_count} imagens")
    else:
        print("\n[FASE 1] Pulando coleta (usando dados existentes)")
    
    # Fase 2: Divisão de dados
    print("\n[FASE 2] Dividindo dados...")
    if not split_data():
        print("[ERRO] Falha na divisão de dados")
        return
    
    # Fase 3: Treinamento
    if not args.skip_train:
        print("\n[FASE 3] Treinando modelo...")
        if not train_model(args.epochs):
            print("[ERRO] Falha no treinamento")
            return
    else:
        print("\n[FASE 3] Treinamento pulado")
    
    print("\n[CONCLUÍDO] Processo finalizado!")

if __name__ == "__main__":
    main()

