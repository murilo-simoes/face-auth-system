FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para OpenCV e outras bibliotecas
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]