"""
Script para dividir dados coletados em train/val/test para treinamento YOLO.
Baseado no código fornecido pelo usuário.
"""
import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/DataCollect"  # Pasta onde os dados foram coletados
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Verificar se pasta de entrada existe
if not os.path.exists(inputFolderPath):
    print(f"[ERRO] Pasta de entrada não encontrada: {inputFolderPath}")
    print(f"[INFO] Execute primeiro o script yolo_data_collect.py para coletar dados")
    exit(1)

# Remover pasta de saída se existir e criar nova
try:
    shutil.rmtree(outputFolderPath)
except OSError:
    pass

# Criar diretórios
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# Obter nomes dos arquivos
listNames = os.listdir(inputFolderPath)
uniqueNames = []

for name in listNames:
    if name.endswith('.jpg'):
        uniqueNames.append(name.split('.')[0])

uniqueNames = list(set(uniqueNames))

if len(uniqueNames) == 0:
    print(f"[ERRO] Nenhuma imagem encontrada em {inputFolderPath}")
    exit(1)

# Embaralhar
random.shuffle(uniqueNames)

# Calcular número de imagens para cada pasta
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# Colocar imagens restantes no treinamento
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# Dividir a lista
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]

print(f'Total de imagens: {lenData}')
print(f'Split: Train={len(Output[0])}, Val={len(Output[1])}, Test={len(Output[2])}')

# Copiar arquivos
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        # Copiar imagem
        srcImg = f'{inputFolderPath}/{fileName}.jpg'
        dstImg = f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg'
        
        # Copiar label
        srcLabel = f'{inputFolderPath}/{fileName}.txt'
        dstLabel = f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt'
        
        if os.path.exists(srcImg):
            shutil.copy(srcImg, dstImg)
        else:
            print(f"[AVISO] Imagem não encontrada: {srcImg}")
        
        if os.path.exists(srcLabel):
            shutil.copy(srcLabel, dstLabel)
        else:
            print(f"[AVISO] Label não encontrado: {srcLabel}")

print("Processo de divisão concluído!")

# Criar arquivo data.yaml
dataYaml = f'''path: ../Data
train: ../train/images
val: ../val/images
test: ../test/images

nc: {len(classes)}
names: {classes}'''

yamlPath = f"{outputFolderPath}/data.yaml"
with open(yamlPath, 'w') as f:
    f.write(dataYaml)

print(f"Arquivo data.yaml criado em: {yamlPath}")
print("\n[PRÓXIMO PASSO] Execute yolo_train.py para treinar o modelo")

