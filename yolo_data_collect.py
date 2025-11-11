"""
Script para coletar dados de faces (fake/real) para treinamento YOLO.
Baseado no código fornecido pelo usuário.
"""
from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import os

####################################
classID = 0  # 0 = fake, 1 = real (altere manualmente antes de coletar)
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35  # Maior = mais foco
debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

# Criar pasta de saída se não existir
os.makedirs(outputFolderPath, exist_ok=True)

# Inicializar câmera
cap = cv2.VideoCapture(0)  # 0 = webcam padrão, 1 = segunda câmera
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

print(f"[INFO] Coletando dados para classe: {'FAKE' if classID == 0 else 'REAL'}")
print(f"[INFO] Pressione 'q' para sair, 's' para salvar manualmente")
print(f"[INFO] Imagens serão salvas automaticamente quando não estiverem borradas")

while True:
    success, img = cap.read()
    if not success:
        print("[ERRO] Não foi possível ler da câmera")
        break
    
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)
    
    listBlur = []  # True/False indicando se as faces estão borradas
    listInfo = []  # Valores normalizados e nome da classe para arquivo de label
    
    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            
            # Verificar score
            if score > confidence:
                # Adicionar offset à face detectada
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)
                
                # Evitar valores abaixo de 0
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if w < 0:
                    w = 0
                if h < 0:
                    h = 0
                
                # Encontrar borrão
                imgFace = img[y:y + h, x:x + w]
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                
                # Normalizar valores
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                
                # Evitar valores acima de 1
                if xcn > 1:
                    xcn = 1
                if ycn > 1:
                    ycn = 1
                if wn > 1:
                    wn = 1
                if hn > 1:
                    hn = 1
                
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                
                # Desenhar
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(
                    imgOut,
                    f'Score: {int(score * 100)}% Blur: {blurValue}',
                    (x, y - 0),
                    scale=2,
                    thickness=3
                )
                
                if debug:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(
                        img,
                        f'Score: {int(score * 100)}% Blur: {blurValue}',
                        (x, y - 0),
                        scale=2,
                        thickness=3
                    )
                
                # Salvar
                if save:
                    if all(listBlur) and listBlur != []:
                        # Salvar imagem
                        timeNow = time()
                        timeNow = str(timeNow).split('.')
                        timeNow = timeNow[0] + timeNow[1]
                        
                        imgPath = f"{outputFolderPath}/{timeNow}.jpg"
                        cv2.imwrite(imgPath, img)
                        
                        # Salvar arquivo de label
                        labelPath = f"{outputFolderPath}/{timeNow}.txt"
                        with open(labelPath, 'a') as f:
                            for info in listInfo:
                                f.write(info)
                        
                        print(f"[SAVED] {imgPath} (blur={blurValue})")
    
    cv2.imshow("Image", imgOut)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Salvar manualmente
        if listInfo:
            timeNow = time()
            timeNow = str(timeNow).split('.')
            timeNow = timeNow[0] + timeNow[1]
            
            imgPath = f"{outputFolderPath}/{timeNow}.jpg"
            cv2.imwrite(imgPath, img)
            
            labelPath = f"{outputFolderPath}/{timeNow}.txt"
            with open(labelPath, 'a') as f:
                for info in listInfo:
                    f.write(info)
            
            print(f"[MANUAL SAVE] {imgPath}")

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Coleta finalizada. Dados salvos em: {outputFolderPath}")

