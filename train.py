import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split

# Определение устройства
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Кастомный датасет
class IDDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]
        attributes = sample["attributes"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, torch.tensor(attributes, dtype=torch.float32)


# Загрузка датасета
def load_dataset(folder):
    samples = []
    print(f"🔍 Загружаем из папки: {folder}")

    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            print(f"⚠️ Пропущено (не папка): {class_folder}")
            continue

        for fname in os.listdir(class_folder):
            if fname.lower().endswith((".jpg", ".jpeg")):
                fpath = os.path.join(class_folder, fname)
                json_path = os.path.splitext(fpath)[0] + ".json"

                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    samples.append({
                        "image_path": fpath,
                        "label": 0 if class_name == "original" else 1,
                        "attributes": [
                            data["portraitReplace"],
                            data["printedCopy"],
                            data["screenReply"]
                        ]
                    })
                    print(f"✅ Найдено изображение: {fpath}")
                else:
                    print(f"⚠️ Нет соответствующего JSON: {json_path}")
            else:
                print(f"⛔ Пропущено (не .jpg): {fname}")

    print(f"📦 Всего загружено: {len(samples)} примеров\n")
    return samples


# Подготовка данных
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка и разделение данных
train_samples = load_dataset("data/train")
val_samples = load_dataset("data/val")

train_loader = DataLoader(IDDataset(train_samples, data_transforms), batch_size=16, shuffle=True)
val_loader = DataLoader(IDDataset(val_samples, data_transforms), batch_size=16, shuffle=False)

# Определение модели
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)  # 2 for classification + 3 for attributes
model = nn.Sequential(
    model,
    nn.Dropout(p=0.5)
)
model = model.to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion_class = nn.CrossEntropyLoss()
criterion_attr = nn.MSELoss()


# Функция обучения
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, attributes in train_loader:
            images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            class_logits = outputs[:, 0:2]  # First two for classification
            attr_preds = outputs[:, 2:]  # Last three for attributes

            loss_class = criterion_class(class_logits, labels)
            loss_attr = criterion_attr(attr_preds, attributes)
            loss = loss_class + loss_attr
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, attributes in val_loader:
                images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)
                outputs = model(images)
                class_logits = outputs[:, 0:2]
                attr_preds = outputs[:, 2:]

                loss_class = criterion_class(class_logits, labels)
                loss_attr = criterion_attr(attr_preds, attributes)
                loss = loss_class + loss_attr
                val_loss += loss.item()

                _, predicted = torch.max(class_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# Обучение модели
train_model(model, train_loader, val_loader, epochs=10)

# Сохранение модели
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_id_classifier.pth")
print("Model saved to models/resnet18_id_classifier.pth")