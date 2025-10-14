# Face Authentication System

Sistema de autenticação facial desenvolvido em Python, capaz de identificar e autenticar usuários automaticamente, atribuindo permissões de acesso de acordo com três níveis de segurança.

---

## 📝 Descrição do Projeto

O **Face Authentication System** é um sistema de controle de acesso baseado em reconhecimento facial. Ele captura a imagem do usuário via webcam, detecta e reconhece rostos usando **OpenCV** e **face_recognition**, e concede acesso com base em níveis predefinidos:

- **Nível 1:** Acesso geral (usuários comuns).
- **Nível 2:** Acesso restrito a diretores de divisões específicas.
- **Nível 3:** Acesso exclusivo ao Ministro do Meio Ambiente.

O sistema utiliza **MongoDB** como banco de dados para armazenar informações dos usuários e suas codificações faciais, permitindo fácil consulta e manutenção.

---

## ⚙️ Funcionalidades

1. Cadastro de usuários com:
   - Nome
   - Nível de acesso (1, 2 ou 3)
   - Codificação facial
2. Reconhecimento facial em tempo real através da webcam
3. Identificação múltipla de rostos simultaneamente
4. Indicador visual:
   - Retângulo verde para usuários com acesso permitido
   - Retângulo vermelho para usuários não reconhecidos
5. Log de usuários identificados (via banco de dados)

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.11+**
- **OpenCV** – captura e manipulação de vídeo
- **face_recognition** – detecção e reconhecimento facial
- **NumPy** – operações numéricas
- **MongoDB** – armazenamento dos dados de usuários e codificações faciais
- **threading** – processamento paralelo para maior fluidez do vídeo

---

## 📁 Estrutura do Projeto

```

face_auth_system/
│
├─ main.py                 # Script principal de reconhecimento facial
├─ cadastro_usuario.py     # Script para cadastrar novos usuários
├─ database.py             # Funções para interagir com o MongoDB
├─ requirements.txt        # Dependências do projeto
└─ README.md               # Este arquivo

```

---

## 💻 Instalação

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd face_auth_system
```

2. Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure o **MongoDB**:

   - Certifique-se de ter um MongoDB rodando localmente ou na nuvem
   - Edite `database.py` com a URI do seu MongoDB

---

## 🚀 Uso

### 1. Cadastrar usuário

```bash
python cadastro_usuario.py
```

O script vai solicitar:

- Nome do usuário
- Nível de acesso (1, 2 ou 3)

Ele captura a imagem via webcam, processa o rosto e salva no banco de dados.

---

### 2. Executar reconhecimento facial

```bash
python main.py
```

- Pressione **`q`** para sair
- Usuários reconhecidos terão retângulos verdes e seus nomes + nível exibidos
- Usuários desconhecidos aparecem em vermelho

---

## ⚡ Melhorias implementadas

- **Threading**: captura de vídeo e reconhecimento facial rodando em paralelo, garantindo fluidez
- **Redução de frame para processamento**: acelera a detecção sem perder precisão visual
- Suporte a **múltiplos rostos simultâneos**
- Sistema preparado para PCs potentes (roda em alta resolução com quase nenhum lag)

---

## 💡 Próximas melhorias

- Integração com logs automáticos no MongoDB
- Acesso baseado em eventos (por exemplo, abertura de portas automatizada)
- Interface gráfica com PyQt ou Tkinter
- Otimização para GPU (CUDA) para reconhecimento em tempo real ainda mais rápido

---

## 📄 Licença

Este projeto é **open-source** e livre para uso acadêmico e experimental.

---
