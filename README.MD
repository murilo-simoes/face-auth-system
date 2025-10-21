# 🧠 Face Recognition API

API de reconhecimento facial desenvolvida em **Python** utilizando **Flask**, **face_recognition** e **MongoDB Atlas**.  
Permite cadastrar rostos e validar se uma imagem enviada pertence a um rosto já cadastrado.

---

## 🚀 Funcionalidades

### 1. Cadastrar rosto

- Endpoint: `POST /register`
- Envia uma imagem em **base64**, nome e nível de acesso.
- A API extrai o encoding facial e armazena no banco de dados (MongoDB).
- Se o rosto já estiver cadastrado, retorna um erro.

### 2. Verificar rosto

- Endpoint: `POST /verify`
- Envia uma imagem em **base64**.
- A API compara com todos os rostos cadastrados e retorna:

  ```json
  {
    "nome": "Murilo",
    "nivel": 2
  }
  ```

- Caso o rosto não seja identificado:

  ```json
  {
    "erro": "Rosto não reconhecido"
  }
  ```

---

## 🧩 Estrutura do Projeto

```
face-recognition-api/
├── app.py               # Código principal da API Flask
├── database.py          # Conexão e funções do MongoDB
├── utils.py             # Funções auxiliares (conversão base64, encoding, etc)
├── requirements.txt     # Dependências do projeto
└── README.md            # Este arquivo
```

---

## ⚙️ Instalação

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/seuusuario/face-recognition-api.git
cd face-recognition-api
```

### 2️⃣ Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 3️⃣ Instalar as dependências

```bash
pip install -r requirements.txt
```

> Se quiser instalar o Flask manualmente:
>
> ```bash
> pip install flask face_recognition opencv-python pymongo numpy
> ```

---

## ☁️ Configurar MongoDB Atlas

1. Crie uma conta em [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Crie um **Cluster gratuito**
3. Copie a **connection string**, exemplo:

   ```
   mongodb+srv://usuario:senha@cluster0.mongodb.net/?retryWrites=true&w=majority
   ```

4. Crie um arquivo `.env` e adicione:

   ```
   MONGO_URI=sua_connection_string
   DB_NAME=face_recognition
   ```

---

## 🧠 Como Enviar Imagens

Envie o conteúdo da imagem em **base64**:

```json
{
  "nome": "Murilo",
  "nivel": 2,
  "imagem_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

---

## 🧪 Testar com cURL ou Postman

### Cadastrar rosto

```bash
curl -X POST http://127.0.0.1:5000/register \
  -H "Content-Type: application/json" \
  -d "{\"nome\": \"Murilo\", \"nivel\": 2, \"imagem_base64\": \"$(base64 imagem.jpg)\"}"
```

### Verificar rosto

```bash
curl -X POST http://127.0.0.1:5000/verify \
  -H "Content-Type: application/json" \
  -d "{\"imagem_base64\": \"$(base64 imagem_teste.jpg)\"}"
```

---

## ▶️ Rodar o Servidor

```bash
python app.py
```

A API estará disponível em:

```
http://127.0.0.1:5000
```

---

## 📦 Exemplo de Resposta

✅ **Rosto reconhecido**

```json
{
  "nome": "Murilo",
  "nivel": 2
}
```

❌ **Rosto não reconhecido**

```json
{
  "erro": "Rosto não reconhecido"
}
```

---

## 📚 Dependências Principais

- **Flask** — Framework web para criar a API
- **face_recognition** — Biblioteca de reconhecimento facial
- **opencv-python** — Manipulação de imagens
- **pymongo** — Conexão com o MongoDB
- **numpy** — Manipulação de arrays
- **python-dotenv** — Leitura de variáveis de ambiente

---
