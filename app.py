from flask import Flask, request, jsonify, Response
import cv2
import base64
import numpy as np
import face_recognition
import os
import logging
from datetime import datetime
from database import salvar_usuario, buscar_todos_encodings, buscar_todos_encodings_com_id, armarzenar_toxicina, procurar_toxina_por_id, atualizar_toxina, remover_toxina, listar_toxinas, buscar_toxinas_por_nivel_maximo, verificar_usuario_nivel_3, buscar_usuario_por_id, remover_usuario
from utils import decode_base64_image, validate_video_file, extract_frames_from_video, save_temp_video, convert_to_mp4
from flask_cors import CORS
from validate import validateToxin
from anti_spoofing import process_anti_spoofing


logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app)

app.logger.setLevel(logging.DEBUG)
@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    nome = data.get("nome")
    nivel = data.get("nivel")
    image_base64 = data.get("imagem_base64")

    if not nome or not nivel or not image_base64:
        return jsonify({"erro": "Campos obrigatórios: nome, nivel, imagem_base64"}), 400

    if nivel == 3 and verificar_usuario_nivel_3():
        return jsonify({"erro": "Já existe um usuário de nível 3 cadastrado."}), 409

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

    usuario_criado = salvar_usuario(nome, nivel, encoding.tolist(), image_base64)
    
    # Remover face_encoding da resposta (dados sensíveis e muito grandes)
    usuario_resposta = {
        "_id": usuario_criado.get("_id"),
        "nome": usuario_criado.get("nome"),
        "nivel": usuario_criado.get("nivel"),
        "imagem_base64": usuario_criado.get("imagem_base64")
    }

    return jsonify({
        "mensagem": f"Usuário {nome} cadastrado com sucesso!",
        "usuario": usuario_resposta
    }), 201


@app.route("/verify", methods=["POST"])
def verify_face():
    """
    Endpoint de verificação com anti-spoofing.
    Aceita vídeo (multipart/form-data) ou imagem (JSON base64) como fallback.
    """
    temp_video_path = None
    video_format = None
    
    try:
        print(f"[DEBUG] Requisição recebida - Method: {request.method}, Content-Type: {request.content_type}")
        print(f"[DEBUG] Files no request: {list(request.files.keys())}")
        print(f"[DEBUG] Form data: {list(request.form.keys())}")
        
        # Prioridade para vídeo
        if 'video' in request.files:
            video_file = request.files['video']
            print(f"[DEBUG] Arquivo de vídeo recebido: {video_file.filename}")

            if video_file.filename == '':
                print("[ERROR 400] Arquivo de vídeo vazio")
                return jsonify({"erro": "Arquivo de vídeo vazio"}), 400

            content_type = video_file.content_type
            print(f"[DEBUG] Content-Type do vídeo: {content_type}")
            if content_type not in ['video/mp4', 'video/webm']:
                print(f"[ERROR 415] Formato não suportado: {content_type}")
                return jsonify({"erro": "Formato de vídeo não suportado. Use video/mp4 ou video/webm"}), 415

            video_format = "mp4" if content_type == "video/mp4" else "webm"
            
            # Salvar temporário
            temp_video_path = save_temp_video(video_file, video_format)
            if not temp_video_path:
                print("[ERROR 500] Falha ao salvar arquivo temporário")
                return jsonify({"erro": "Erro ao processar vídeo"}), 500

            print(f"[DEBUG] Vídeo salvo temporariamente em: {temp_video_path}")

            # Conversão WebM → MP4
            if video_format == "webm":
                print("[DEBUG] Convertendo WebM para MP4...")
                try:
                    temp_video_path_mp4 = convert_to_mp4(temp_video_path)
                    os.unlink(temp_video_path)
                    temp_video_path = temp_video_path_mp4
                    video_format = "mp4"
                    print(f"[DEBUG] Conversão concluída: {temp_video_path}")
                except Exception as e:
                    print(f"[ERROR 500] Falha na conversão: {e}")
                    raise

            # Validar vídeo
            is_valid, error_msg = validate_video_file(temp_video_path, max_size_mb=15)
            if not is_valid:
                print(f"[ERROR 400] Validação de vídeo falhou: {error_msg}")
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                return jsonify({"erro": error_msg}), 400

            # Extrair frames
            try:
                frames, fps = extract_frames_from_video(temp_video_path, num_frames=12)
                print(f"[DEBUG] Frames extraídos: {len(frames)}, FPS: {fps}")
                if len(frames) < 5:
                    raise ValueError(f"Frames insuficientes: {len(frames)}")
            except Exception as e:
                print(f"[ERROR 500] Falha ao extrair frames: {e}")
                raise

            # Anti-spoofing
            try:
                anti_spoof_result = process_anti_spoofing(frames, fps)
                print(f"[DEBUG] Resultado anti-spoofing: {anti_spoof_result}")
            except Exception as e:
                print(f"[ERROR 500] Falha no anti-spoofing: {e}")
                raise

            # Limpar temporário
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
                temp_video_path = None

            if not anti_spoof_result.get("liveness", False):
                return jsonify({
                    "erro": f"Falha na verificação de liveness: {anti_spoof_result.get('reason', 'spoof_detectado')}",
                    "scores": anti_spoof_result.get("scores", {}),
                    "final_score": anti_spoof_result.get("final_score", 0.0)
                }), 403

            rgb = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)

        else:
            # Fallback imagem base64
            data = request.get_json()
            if not data:
                print("[ERROR 400] Nenhum dado JSON recebido e nenhum vídeo no multipart")
                return jsonify({"erro": "Campo obrigatório: video (multipart) ou imagem_base64 (JSON)"}), 400

            image_base64 = data.get("imagem_base64")
            if not image_base64:
                print("[ERROR 400] Campo imagem_base64 não encontrado no JSON")
                return jsonify({"erro": "Campo obrigatório: video (multipart) ou imagem_base64 (JSON)"}), 400

            rgb = decode_base64_image(image_base64)
            if rgb is None:
                print("[ERROR 400] Falha ao decodificar imagem base64")
                return jsonify({"erro": "Imagem inválida"}), 400

            anti_spoof_result = {"liveness": True, "final_score": 1.0, "scores": {"yolo": 1.0}, "reason": "image_fallback"}

        # Reconhecimento facial
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            print("[ERROR 400] Nenhum rosto detectado na imagem/frame")
            return jsonify({"erro": "Nenhum rosto detectado"}), 400

        encoding = np.array(encodings[0])
        usuarios = buscar_todos_encodings_com_id()
        if not usuarios:
            return jsonify({"erro": "Nenhum usuário cadastrado"}), 404

        best_match = None
        lowest_distance = 1.0
        for u in usuarios:
            known_encoding = np.array(u["face_encoding"])
            distance = face_recognition.face_distance([known_encoding], encoding)[0]
            if distance < lowest_distance and distance < 0.45:
                lowest_distance = distance
                best_match = u

        if best_match:
            return jsonify({
                "_id": best_match.get("_id"),
                "nome": best_match["nome"],
                "nivel": best_match["nivel"],
                "imagem_base64": best_match.get("imagem_base64")
            }), 200

        return jsonify({"erro": "Rosto não reconhecido"}), 404

    except Exception as e:
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        print(f"[CRITICAL] Exceção não tratada em verify_face: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            "liveness": False,
            "final_score": 0.0,
            "scores": {"yolo": 0.0},
            "identity_match": False,
            "identity_confidence": 0.0,
            "reason": "erro_processamento",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 500


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

@app.get("/toxin/user/<string:id>")
def list_toxins_by_user_id(id: str):
    try:
        usuario = buscar_usuario_por_id(id)
        if usuario is None:
            return jsonify({
                "erro": "Usuário não encontrado"
            }), 404
        
        nivel_usuario = usuario.get("nivel")
        toxinas = buscar_toxinas_por_nivel_maximo(nivel_usuario)
        return jsonify(toxinas), 200
    
    except Exception as e:
        print(e)
        return jsonify({
            "erro": "Não foi possível buscar as toxinas no momento."
        }), 500


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

@app.delete("/user/<string:id>")
def delete_user(id: str):
    try:
        usuario = buscar_usuario_por_id(id)
        if usuario is None:
            return jsonify({
                "erro": "Usuário não encontrado"
            }), 404

        remover_usuario(id)
        return jsonify({
            "mensagem": f"Usuário {usuario.get('nome')} removido com sucesso!"
        }), 200
    
    except Exception as e:
        print(e)
        return jsonify({
            "erro": "Não foi possível remover o usuário no momento."
        }), 500

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)


