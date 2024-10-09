import sys
import os

# Añade la ruta de la raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from huggingface_hub import hf_hub_download
from google.colab import drive
import json
from CRM import CRM  # Asegúrate de definir tu modelo CRM

# Añadir la ruta del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('/content/drive/MyDrive/data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.ImageFolder('/content/drive/MyDrive/data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Descargar pesos preentrenados
model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

specs = json.load(open("/content/finetuning/configs/specs_objaverse_total.json"))

# Cargar el modelo en el dispositivo adecuado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRM(specs).to(device)
model.train() 

# Cargar los pesos
model.load_state_dict(torch.load(model_path, map_location=device))
print("Pesos cargados con éxito.")
model = model.to(device)
# # Congelar capas
# for param in model.parameters():
#     param.requires_grad = False

# for param in model.decoder.parameters():
#     param.requires_grad = True

# # Configurar el optimizador y la función de pérdida
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Revisar parámetros antes de pasar al optimizador
params = list(filter(lambda p: p.requires_grad, model.parameters()))
print(f"Número de parámetros con gradiente: {len(params)}")

# Inicializar el optimizador solo si hay parámetros con gradiente
if len(params) > 0:
    optimizer = optim.Adam(params, lr=0.0001)
else:
    raise ValueError("No hay parámetros con gradientes para optimizar.")
criterion = nn.CrossEntropyLoss()

num_epochs = 10

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Chequeo de formas
        print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')

        optimizer.zero_grad()
        outputs = model(inputs)

        # Chequeo de salidas
        if outputs is None or outputs.shape[0] != targets.shape[0]:
            raise ValueError("El modelo no está generando salidas válidas.")

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1}] pérdida: {running_loss / 10:.3f}')
            running_loss = 0.0

# Evaluación
model.eval()
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f'Pérdida en el conjunto de prueba: {test_loss / len(test_loader):.3f}')

# Guardar el modelo
os.makedirs('checkpoints', exist_ok=True)  # Asegurarse de que el directorio exista
torch.save(model.state_dict(), 'checkpoints/modelo_finetune_gafas.pth')
