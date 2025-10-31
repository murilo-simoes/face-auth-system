from flask import Flask, request, jsonify, Response
import cv2
import base64
import numpy as np
import face_recognition
from database import salvar_usuario, buscar_todos_encodings, armarzenar_toxicina, procurar_toxina_por_id, atualizar_toxina, remover_toxina, listar_toxinas
from utils import decode_base64_image 
from flask_cors import CORS
from validate import validateToxin

app = Flask(__name__)
CORS(app)

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

    salvar_usuario(nome, nivel, encoding.tolist(), image_base64)

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
            "nivel": best_match["nivel"],
            "imagem_base64": best_match.get("imagem_base64") 
        }), 200
    else:
        return jsonify({"erro": "Rosto não reconhecido"}), 404

@app.get("/toxin")
def list_all_toxins():
    params = {
        "nome": request.args.get('nome'),
        "tipo": request.args.get('tipo'),
        "periculosidade": request.args.get('periculosidade'),
        "nivel": request.args.get('nivel')
    }

    toxins = listar_toxinas(params)
    return jsonify(toxins), 200


@app.post('/toxin')
def store_toxin():
    body = request.get_json()

    toxin = {
        "nome": body.get('nome'),
        "tipo": body.get('tipo'),
        "periculosidade": body.get('periculosidade'),
        "nivel": body.get('nivel')
    }

    if not validateToxin(toxin):
        return jsonify({
            "error": "Invalid data",
        }), 400
    
    try:
        armarzenar_toxicina(toxin)
        return jsonify({
            "message": "toxin store successfully!"
        }), 201
    except Exception as e:
        print(e)
        return jsonify({
            "message": "Can't store data at the moment"
        }), 500

@app.put("/toxin/<string:id>")
def update_toxin(id: str):
    try:
        if procurar_toxina_por_id(id) is None:
            return jsonify({
                "error": "Toxin not found"
            }), 404
        
        body = request.get_json()

        toxin = {
            "nome": body.get('nome'),
            "tipo": body.get('tipo'),
            "periculosidade": body.get('periculosidade'),
            "nivel": body.get('nivel')
        }

        if not validateToxin(toxin):
            return jsonify({
                "error": "invalid data"
            }), 400

    
        atualizar_toxina(id, toxin)
        return Response(status=204)
    
    except:
        return jsonify({
            "error": "Can't update toxin"
        }), 500

@app.delete("/toxin/<string:id>")
def delete_toxin(id: str):
    try:
        if procurar_toxina_por_id(id) is None:
            return jsonify({
                "error": "Toxina not found"
            }), 404

        remover_toxina(id)
        return Response(status=204)
    
    except:
        return jsonify({
            "error": "Can't remove toxin at the moment."
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

