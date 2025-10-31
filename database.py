import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import certifi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "face_auth")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "users")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def salvar_usuario(nome: str, nivel: int, face_encoding: list, imagem_base64: str):
    collection.insert_one({
        "nome": nome,
        "nivel": nivel,
        "face_encoding": face_encoding,
        "imagem_base64": imagem_base64
    })


def buscar_todos_encodings():
    usuarios = list(collection.find({}, {"_id": 0}))
    return usuarios


def buscar_por_nome(nome: str):
    return collection.find_one({"nome": nome}, {"_id": 0})

def armarzenar_toxicina(toxin: dict) -> None:
    criado_em = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    db.toxin.insert_one({
        **toxin,
        "criado_em": criado_em
    })
    