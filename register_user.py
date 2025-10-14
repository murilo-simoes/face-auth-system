import cv2
import face_recognition
from database import salvar_usuario

nome = input("Nome do usuário: ")
nivel = int(input("Nível de acesso (1, 2, 3): "))

print("Capturando imagem da câmera...")
video = cv2.VideoCapture(0)

for i in range(30):
    ret, frame = video.read()

video.release()

if not ret:
    print("❌ Erro ao capturar imagem da câmera.")
    exit()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
encodings = face_recognition.face_encodings(rgb)

if encodings:
    encoding = encodings[0].tolist()
    salvar_usuario(nome, nivel, encoding)
    print(f"✅ Usuário {nome} (nível {nivel}) cadastrado com sucesso!")
else:
    print("❌ Nenhum rosto detectado. Tente novamente.")
