import cv2
import face_recognition
import numpy as np
from database import buscar_todos_encodings

usuarios = buscar_todos_encodings()
encodings_conhecidos = [u["face_encoding"] for u in usuarios]
nomes_conhecidos = [u["nome"] for u in usuarios]
niveis_conhecidos = [u["nivel"] for u in usuarios]

video = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

process_this_frame = True

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        face_levels = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings_conhecidos, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(encodings_conhecidos, face_encoding)
            name = "Desconhecido"
            nivel = 0
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = nomes_conhecidos[best_match_index]
                nivel = niveis_conhecidos[best_match_index]
            face_names.append(name)
            face_levels.append(nivel)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name, nivel in zip(face_locations, face_names, face_levels):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        color = (0, 255, 0) if nivel > 0 else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} (Nível {nivel})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento Facial - Sistema de Acesso", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
