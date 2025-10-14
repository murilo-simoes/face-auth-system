from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
usuarios = db["users"]

def salvar_usuario(nome, nivel, encoding):
    usuarios.insert_one({
        "nome": nome,
        "nivel": nivel,
        "face_encoding": encoding,
    })

def listar_usuarios():
    return list(usuarios.find({}, {"_id": 0}))

def buscar_todos_encodings():
    data = list(usuarios.find({}, {"_id": 0, "face_encoding": 1, "nome": 1, "nivel": 1}))
    return data
