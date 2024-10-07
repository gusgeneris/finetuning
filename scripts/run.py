# import sys
# import os

# # Añade la ruta de la raíz del proyecto al sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# from huggingface_hub import hf_hub_download
# from google.colab import drive
# import json
# from CRM import CRM  # Asegúrate de definir tu modelo CRM

# # Definir transformaciones para las imágenes
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Ajusta según el modelo CRM
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_data = datasets.ImageFolder('/content/drive/MyDrive/data/train', transform=transform)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# test_data = datasets.ImageFolder('/content/drive/MyDrive/data/test', transform=transform)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# # Descargar los pesos preentrenados del modelo CRM desde Hugging Face Hub
# model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

# # specs = json.load(open("configs/specs_objaverse_total.json"))
# specs = json.load(open("/content/finetuning/configs/specs_objaverse_total.json"))

# # Cargar el modelo preentrenado CRM (asegúrate de definir la clase CRM antes de cargar los pesos)
# model = CRM(specs).to("cuda")  # Definir correctamente la clase CRM
# model.train() 

# # Mover el modelo a la GPU antes de cargar los pesos
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Asegúrate de que cada submódulo esté en el dispositivo correcto
# model = model.to(device)
# for name, module in model.named_modules():
#     # Solo imprime si el módulo tiene parámetros
#     if len(list(module.parameters())) > 0:
#         module.to(device)
#         print(f'{name} is on device: {next(module.parameters()).device}')
#     else:
#         print(f'{name} does not have parameters.')

# # Cargar los pesos preentrenados en el mismo dispositivo que el modelo
# model.load_state_dict(torch.load(model_path, map_location=device))
# print("Pesos cargados con éxito.")

# # Asegúrate de que todas las capas del modelo también estén en el mismo dispositivo
# model = model.to(device)
# for name, module in model.named_modules():
#     module.to(device)  

# # Congelar las capas iniciales del modelo para mantener las características aprendidas
# for param in model.parameters():
#     param.requires_grad = True

# # Descongelar las últimas capas para ajustarlas a las características de gafas
# for param in model.decoder.parameters():  # Ajusta según las capas finales de tu modelo CRM
#     param.requires_grad = True

# # for name, param in model.named_parameters():
# #     print(name, param.requires_grad)
# #     print('params')


# # Configurar el optimizador
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# # Función de pérdida (ajustar si es necesario según la tarea)
# # Usar CrossEntropyLoss si es una tarea de clasificación
# criterion = nn.CrossEntropyLoss()

# num_epochs = 10          

# # Entrenamiento del modelo
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # Obtener inputs y targets del loader
#         inputs, targets = data
        
#         # Mover los inputs y targets al dispositivo (CPU o GPU)
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Verificar la forma de los inputs y targets
#         print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')

#         print(f"Input tensor device: {inputs.device}")
#         print(f"UNet2 weights device: {next(model.unet2.parameters()).device}")
#         inputs = inputs.to(device)


#         # Poner a cero los gradientes del optimizador
#         optimizer.zero_grad()

#         # print(inputs)
#         # Pasar los inputs al modelo
#         outputs = model(inputs) 
#         print(outputs)
#         if outputs is None:
#             raise ValueError("El modelo no está generando salidas válidas.")

#         # Calcular la pérdida
#         loss = criterion(outputs, targets)

#         # Backpropagation (gradientes)
#         loss.backward()

#         # Actualizar los parámetros
#         optimizer.step()

#         # Acumular la pérdida para monitoreo
#         running_loss += loss.item()
        
#         # Mostrar la pérdida cada 10 lotes
#         if i % 10 == 9:
#             print(f'[{epoch + 1}, {i + 1}] pérdida: {running_loss / 10:.3f}')
#             running_loss = 0.0


# # Evaluación en el conjunto de prueba
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, targets = data
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         test_loss += loss.item()

# print(f'Pérdida en el conjunto de prueba: {test_loss / len(test_loader):.3f}')

# # Guardar el modelo ajustado
# torch.save(model.state_dict(), 'checkpoints/modelo_finetune_gafas.pth')


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
import json
from CRM import CRM  # Asegúrate de definir tu modelo CRM

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajusta según el modelo CRM
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definir datasets y loaders
train_data = datasets.ImageFolder('/content/drive/MyDrive/data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.ImageFolder('/content/drive/MyDrive/data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Descargar los pesos preentrenados del modelo CRM desde Hugging Face Hub
model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

# Cargar especificaciones
specs = json.load(open("/content/finetuning/configs/specs_objaverse_total.json"))

# Definir el dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo preentrenado CRM
model = CRM(specs).to(device)

# Verificación de que todas las capas y submódulos del modelo estén en la GPU
for name, module in model.named_modules():
    if len(list(module.parameters())) > 0:
        module.to(device)
        print(f'{name} is on device: {next(module.parameters()).device}')
    else:
        print(f'{name} does not have parameters.')

# Cargar los pesos preentrenados en el modelo
model.load_state_dict(torch.load(model_path, map_location=device))
print("Pesos cargados con éxito.")

# Verificar que todas las partes clave del modelo están en la GPU
model.unet2 = model.unet2.to(device)  # Mover unet2 explícitamente
print(f"UNet2 weights device: {next(model.unet2.parameters()).device}")

# Congelar capas iniciales del modelo, si lo deseas
for param in model.parameters():
    param.requires_grad = True  # Descongelar todas las capas

# Descongelar capas específicas para ajustar a gafas (si es necesario)
for param in model.decoder.parameters():
    param.requires_grad = True

# Configurar el optimizador
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Definir la función de pérdida (CrossEntropyLoss es común para clasificación)
criterion = nn.CrossEntropyLoss()

# Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        
        # Mover inputs y targets al dispositivo
        inputs, targets = inputs.to(device), targets.to(device)

        # Verificar las formas de los tensores
        print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')
        print(f"Input tensor device: {inputs.device}")
        print(f"UNet2 weights device: {next(model.unet2.parameters()).device}")

        # Si hay que cambiar el tamaño de los inputs, asegúrate de que todo esté en el dispositivo
        inputs = torch.nn.functional.interpolate(inputs, size=(256, 256)).to(device)
        print(f"Input shape after resize: {inputs.shape}")

        # Zero gradientes
        optimizer.zero_grad()

        # Pasar los inputs por el modelo
        outputs = model(inputs)
        if outputs is None:
            raise ValueError("El modelo no está generando salidas válidas.")
        
        # Calcular la pérdida
        loss = criterion(outputs, targets)

        # Backpropagation y actualización de los pesos
        loss.backward()
        optimizer.step()

        # Acumular la pérdida
        running_loss += loss.item()

        # Mostrar la pérdida cada 10 lotes
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1}] pérdida: {running_loss / 10:.3f}')
            running_loss = 0.0

# Evaluación en el conjunto de prueba
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

# Guardar el modelo ajustado
torch.save(model.state_dict(), 'checkpoints/modelo_finetune_gafas.pth')
