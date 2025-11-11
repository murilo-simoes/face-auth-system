"""
Script para treinar modelo YOLO para detecção de spoofing.
Baseado no código fornecido pelo usuário.
"""
from ultralytics import YOLO
import os

# Configurações
model_name = 'yolov8n.pt'  # Modelo base YOLO (nano - mais rápido)
data_yaml = 'Dataset/SplitData/data.yaml'  # Caminho para data.yaml
epochs = 50  # Número de épocas (aumente para melhor precisão)
output_dir = 'models'  # Pasta onde salvar o modelo treinado

# Verificar se data.yaml existe
if not os.path.exists(data_yaml):
    print(f"[ERRO] Arquivo data.yaml não encontrado: {data_yaml}")
    print(f"[INFO] Execute primeiro yolo_split_data.py para criar o data.yaml")
    exit(1)

# Verificar se pasta de dados existe
split_data_path = os.path.dirname(data_yaml)
if not os.path.exists(split_data_path):
    print(f"[ERRO] Pasta de dados não encontrada: {split_data_path}")
    print(f"[INFO] Execute primeiro yolo_split_data.py para dividir os dados")
    exit(1)

# Criar pasta de modelos se não existir
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Iniciando treinamento YOLO...")
print(f"[INFO] Modelo base: {model_name}")
print(f"[INFO] Data YAML: {data_yaml}")
print(f"[INFO] Épocas: {epochs}")
print(f"[INFO] Pasta de saída: {output_dir}")
print(f"[INFO] Isso pode levar algum tempo...")

# Carregar modelo YOLO
model = YOLO(model_name)

# Treinar modelo
try:
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,  # Tamanho da imagem
        batch=16,  # Tamanho do batch (ajuste conforme sua GPU/RAM)
        name='anti_spoofing_yolo',  # Nome do experimento
        project='runs/detect',  # Pasta de resultados
        save=True,
        val=True,  # Validar durante treinamento
    )
    
    # Salvar modelo final
    best_model_path = f"{output_dir}/anti_spoofing_yolo.pt"
    model.save(best_model_path)
    
    print(f"\n[SUCESSO] Treinamento concluído!")
    print(f"[INFO] Modelo salvo em: {best_model_path}")
    print(f"\n[PRÓXIMO PASSO] Configure no .env:")
    print(f"YOLO_MODEL_PATH={best_model_path}")
    
except Exception as e:
    print(f"[ERRO] Falha no treinamento: {e}")
    exit(1)

