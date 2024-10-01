import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from huggingface_hub import hf_hub_download
from google.colab import drive
from CRM import CRM  # Asegúrate de definir tu modelo CRM

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajusta según el modelo CRM
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('/content/drive/MyDrive/data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.ImageFolder('/content/drive/MyDrive/data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Descargar los pesos preentrenados del modelo CRM desde Hugging Face Hub
model_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")

# Cargar el modelo preentrenado CRM (asegúrate de definir la clase CRM antes de cargar los pesos)
model = CRM()  # Definir correctamente la clase CRM
model.load_state_dict(torch.load(model_path))

# Congelar las capas iniciales del modelo para mantener las características aprendidas
for param in model.parameters():
    param.requires_grad = False

# Descongelar las últimas capas para ajustarlas a las características de gafas
for param in model.decoder.parameters():  # Ajusta según las capas finales de tu modelo CRM
    param.requires_grad = True

# Configurar el optimizador
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Función de pérdida (ajustar si es necesario según la tarea)
# Usar CrossEntropyLoss si es una tarea de clasificación
criterion = nn.CrossEntropyLoss()

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
