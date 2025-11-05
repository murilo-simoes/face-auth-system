import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson.regex import Regex
from re import compile
from bson.objectid import ObjectId
from typing import Union
import certifi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "face_auth")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "users")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def salvar_usuario(nome: str, nivel: int, face_encoding: list, imagem_base64: str) -> dict:
    """
    Salva um novo usuário e retorna o usuário criado com o ID.
    """
    result = collection.insert_one({
        "nome": nome,
        "nivel": nivel,
        "face_encoding": face_encoding,
        "imagem_base64": imagem_base64
    })
    usuario = collection.find_one({"_id": result.inserted_id})
    if usuario:
        usuario['_id'] = str(usuario['_id'])
    return usuario


def buscar_todos_encodings():
    usuarios = list(collection.find({}, {"_id": 0}))
    return usuarios

def buscar_todos_encodings_com_id():
    """
    Busca todos os usuários com seus IDs incluídos.
    """
    usuarios = list(collection.find({}))
    for usuario in usuarios:
        usuario['_id'] = str(usuario['_id'])
    return usuarios


def buscar_por_nome(nome: str):
    return collection.find_one({"nome": nome}, {"_id": 0})

def buscar_usuario_por_nome(nome: str):
    """
    Busca um usuário por nome e retorna com _id.
    """
    return collection.find_one({"nome": nome})

def buscar_usuario_por_id(id: str) -> Union[dict, None]:
    """
    Busca um usuário por ID do MongoDB.
    """
    try:
        usuario = collection.find_one({"_id": ObjectId(id)})
        if usuario:
            usuario['_id'] = str(usuario['_id'])
        return usuario
    except:
        return None

def remover_usuario(id: str) -> bool:
    """
    Remove um usuário pelo ID do MongoDB.
    Retorna True se o usuário foi removido, False caso contrário.
    """
    try:
        result = collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0
    except:
        return False

def verificar_usuario_nivel_3():
    """
    Verifica se já existe um usuário de nível 3 cadastrado.
    Retorna True se existir, False caso contrário.
    """
    usuario = collection.find_one({"nivel": 3})
    return usuario is not None

def armarzenar_toxicina(toxin: dict) -> None:
    criado_em = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    db.toxin.insert_one({
        **toxin,
        "criado_em": criado_em
    })
    

def procurar_toxina_por_id(id: str) -> Union[dict, None]:
    return db.toxin.find_one({
        "_id": ObjectId(id)
    })

def atualizar_toxina(id: str, toxina: dict) -> None:
    db.toxin.update_one(
        {
            "_id": ObjectId(id)
        },
        {
            "$set": {
                **toxina
            }
        }
    )

def remover_toxina(id: str) -> None:
    db.toxin.delete_one({
        "_id": ObjectId(id)
    })

def listar_toxinas(params: dict) -> list[dict]:
    numeric_fields = ['nivel', 'periculosidade']

    query = {}

    for field in params:
        if params[field] is None:
            continue

        if field in numeric_fields:
            query[field] = int(params[field])
            continue

        query[field] = Regex.from_native(compile(f".*{params[field]}.*"))


    toxins = db.toxin.find(query).to_list()

    for i in range(len(toxins)):
        toxins[i]['_id'] = str(toxins[i]['_id'])

    return toxins

def buscar_toxinas_por_nivel_maximo(nivel_maximo: int) -> list[dict]:
    """
    Busca toxinas com nível menor ou igual ao nível máximo fornecido.
    """
    toxins = list(db.toxin.find({"nivel": {"$lte": nivel_maximo}}))
    
    for i in range(len(toxins)):
        toxins[i]['_id'] = str(toxins[i]['_id'])
    
    return toxins