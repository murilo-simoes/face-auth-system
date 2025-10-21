from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import face_recognition
from database import salvar_usuario, buscar_todos_encodings
from utils import decode_base64_image 

app = Flask(__name__)


@app.route("/register", methods=["POST"])
def register_face():
 
    data = request.get_json()
    nome = data.get("nome")
    nivel = data.get("nivel")
    image_base64 = data.get("imagem_base64")

    if not nome or not nivel or not image_base64:
        return jsonify({"erro": "Campos obrigatórios: nome, nivel, imagem_base64"}), 400

    rgb = decode_base64_image(image_base64)
    if rgb is None:
        return jsonify({"erro": "Imagem inválida"}), 400

    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return jsonify({"erro": "Nenhum rosto detectado"}), 400

    encoding = encodings[0]

    usuarios = buscar_todos_encodings()
    for u in usuarios:
        known_encoding = np.array(u["face_encoding"])
        match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.45)[0]
        if match:
            return jsonify({"erro": f"O rosto já está cadastrado como {u['nome']}"}), 409

    salvar_usuario(nome, nivel, encoding.tolist())
    return jsonify({"mensagem": f"Usuário {nome} cadastrado com sucesso!", "nivel": nivel}), 201


@app.route("/verify", methods=["POST"])
def verify_face():
    data = request.get_json()
    image_base64 = data.get("imagem_base64")

    if not image_base64:
        return jsonify({"erro": "Campo obrigatório: imagem_base64"}), 400

    rgb = decode_base64_image(image_base64)
    if rgb is None:
        return jsonify({"erro": "Imagem inválida"}), 400

    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return jsonify({"erro": "Nenhum rosto detectado"}), 400

    encoding = np.array(encodings[0])

    usuarios = buscar_todos_encodings()
    if not usuarios:
        return jsonify({"erro": "Nenhum usuário cadastrado"}), 404

    best_match = None
    lowest_distance = 1.0

    for u in usuarios:
        known_encoding = np.array(u["face_encoding"])
        distance = face_recognition.face_distance([known_encoding], encoding)[0]
        if distance < lowest_distance and distance < 0.45:  # tolerância ajustável
            lowest_distance = distance
            best_match = u

    if best_match:
        return jsonify({
            "nome": best_match["nome"],
            "nivel": best_match["nivel"]
        }), 200
    else:
        return jsonify({"erro": "Rosto não reconhecido"}), 404


if __name__ == "__main__":
    app.run(debug=True)
