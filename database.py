import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "face_auth")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "users")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def salvar_usuario(nome: str, nivel: int, face_encoding: list):
    collection.insert_one({
        "nome": nome,
        "nivel": nivel,
        "face_encoding": face_encoding
    })


def buscar_todos_encodings():
    usuarios = list(collection.find({}, {"_id": 0}))
    return usuarios


def buscar_por_nome(nome: str):
    return collection.find_one({"nome": nome}, {"_id": 0})
