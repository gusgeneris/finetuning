# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import os
# from huggingface_hub import hf_hub_download

# # Obtener la ruta absoluta del directorio base del proyecto
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Definir las rutas de los datos y del modelo
# train_dir = os.path.join(base_dir, 'data/train')
# test_dir = os.path.join(base_dir, 'data/test')
# # model_path = os.path.join(base_dir, 'model/CRM.pth')
# # model_path = '/home/gustavo/Documentos/Materias/proyecto_finetuning/model/CRM'
# model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

# # Verificar si el modelo existe antes de cargarlo
# if not os.path.isfile(model_path):
#     raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
# else:
#     print(f"Modelo encontrado en: {model_path}")


# # Verificar que las rutas de datos existen
# if not os.path.exists(train_dir):
#     raise FileNotFoundError(f"No se encontró la carpeta de entrenamiento en: {train_dir}")
# if not os.path.exists(test_dir):
#     raise FileNotFoundError(f"No se encontró la carpeta de prueba en: {test_dir}")

# # Definir transformaciones para las imágenes
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Cargar los datos de entrenamiento y prueba
# train_data = datasets.ImageFolder(train_dir, transform=transform)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# test_data = datasets.ImageFolder(test_dir, transform=transform)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# # Verificar si las clases se detectan correctamente
# print(f'Clases detectadas: {train_data.classes}')

# # Cargar el modelo preentrenado
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

# model = torch.load(model_path)

# # Congelar las capas iniciales del modelo
# for param in model.parameters():
#     param.requires_grad = False

# # Descongelar las últimas capas
# for param in model.layer4.parameters():  # Cambia 'layer4' según el modelo
#     param.requires_grad = True

# # Configurar el optimizador
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# # Definir la función de pérdida
# criterion = nn.CrossEntropyLoss()

# # Entrenamiento del modelo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, targets = data
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 10 == 9:  # Cada 10 lotes
#             print(f'[{epoch + 1}, {i + 1}] pérdida: {running_loss / 10:.3f}')
#             running_loss = 0.0

# # Evaluación en el conjunto de prueba
# model.eval()
# test_loss = 0.0
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, targets = data
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         test_loss += loss.item()

#         # Predicciones correctas
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()

# # Mostrar la precisión
# accuracy = 100 * correct / total
# print(f'Pérdida en el conjunto de prueba: {test_loss / len(test_loader):.3f}')
# print(f'Precisión en el conjunto de prueba: {accuracy:.2f}%')

# # Guardar el modelo ajustado
# checkpoint_dir = os.path.join(base_dir, 'checkpoints')
# os.makedirs(checkpoint_dir, exist_ok=True)
# torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'modelo_finetune_gafas.pth'))


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from huggingface_hub import hf_hub_download
from google.colab import drive

# drive.mount('/content/drive')
# # Ver el contenido de la carpeta 'data' para asegurarse de que las subcarpetas 'train' y 'test' existen
# print("Contenido de la carpeta 'data':")
# print(os.listdir('data'))

# # Ver el contenido de la carpeta 'train'
# print("\nContenido de 'data/train':")
# print(os.listdir('data/train'))

# # Ver el contenido de la carpeta 'test'
# print("\nContenido de 'data/test':")
# print(os.listdir('data/test'))


# Cargar los datos de entrenamiento y prueba
# train_data = datasets.ImageFolder('data/train', transform=transform)



# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('/content/drive/MyDrive/data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# test_data = datasets.ImageFolder('data/test', transform=transform)

train_data = datasets.ImageFolder('/content/drive/MyDrive/data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Descargar el modelo desde Hugging Face Hub
model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

# Cargar los datos de entrenamiento y prueba
train_data = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Cargar el modelo preentrenado
model = torch.load(model_path)

# Congelar las capas iniciales del modelo
for param in model.parameters():
    param.requires_grad = False

# Descongelar las últimas capas
for param in model.layer4.parameters():  # Ajusta 'layer4' según el modelo
    param.requires_grad = True

# Configurar el optimizador
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Definir la función de pérdida
criterion = nn.MSELoss()

# Entrenamiento del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
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
