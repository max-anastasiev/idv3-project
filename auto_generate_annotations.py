import cv2
import json
import os
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift
import torch
from torchvision import models, transforms
from torch.nn import functional as F

# Путь к файлам модели для детекции лиц (ResNet-10 SSD)
FACE_PROTO = "/Users/macbookair/PycharmProjects/idv3/opencv_face_detector.pbtxt"
FACE_MODEL = "/Users/macbookair/PycharmProjects/idv3/opencv_face_detector_uint8.pb"

# Load pre-trained screen artifact detection model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
screen_model = models.resnet18(pretrained=True)
screen_model.fc = torch.nn.Linear(screen_model.fc.in_features, 2)  # Binary classification
screen_model.load_state_dict(torch.load("models/screen_artifact_classifier.pth", map_location=device))  # Train and save this model
screen_model.eval()
screen_model.to(device)

# Image transform for screen model
screen_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_face_detector():
    """Загрузка модели детекции лиц."""
    try:
        net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
        return net
    except Exception as e:
        print(f"Ошибка загрузки модели детекции лиц: {e}")
        return None

def detect_face(image, net, confidence_threshold=0.7):
    """Детекция лица на изображении."""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            if endX > startX and endY > startY:
                return (startX, startY, endX, endY), confidence
    return None, 0.0

def compute_portrait_replace(image, face_box, confidence):
    """Оценка вероятности замены портрета с использованием LBP."""
    if face_box is None or confidence < 0.7:
        return 0.0
    (startX, startY, endX, endY) = face_box
    face_region = image[startY:endY, startX:endX]
    if face_region.size == 0:
        return 0.0
    gray_face = cv2.GaussianBlur(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    radius, n_points = 2, 8 * 2
    lbp = local_binary_pattern(gray_face, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    texture_score = 1.0 - np.var(lbp_hist)
    portrait_replace = min(1.0 - (texture_score * (1.0 - confidence)), 1.0)
    return portrait_replace

def compute_printed_copy(image):
    """Оценка вероятности печатной копии с использованием Canny и Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    max_laplacian = 800
    sharpness_score = min(laplacian_var / max_laplacian, 1.0)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges == 255) / (gray.shape[0] * gray.shape[1])
    max_edge_ratio = 0.05
    edge_score = min(edge_ratio / max_edge_ratio, 1.0)
    mean_color_var = np.var(cv2.meanStdDev(gray)[0])
    color_uniformity = min(mean_color_var / 1000, 1.0)
    printed_copy = max(1.0 - sharpness_score, edge_score, color_uniformity)
    return printed_copy

def compute_screen_reply(image):
    """Оценка вероятности скриншота с использованием обученной модели."""
    # Preprocess image
    input_tensor = screen_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = screen_model(input_tensor)
        probs = F.softmax(output, dim=1)
        screen_prob = probs[0, 1].item()  # Probability of screen artifact (class 1)
    return screen_prob

def save_annotation(image_path, portrait_replace, printed_copy, screen_reply):
    """Сохранение аннотаций в JSON формат."""
    annotation_path = os.path.splitext(image_path)[0] + ".json"
    annotation = {
        "image_name": os.path.basename(image_path),
        "portraitReplace": float(portrait_replace),
        "printedCopy": float(printed_copy),
        "screenReply": float(screen_reply)
    }
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=4)
    print(f"Аннотация сохранена для {image_path}")

def generate_annotations_for_folder(folder_path):
    """Генерируем аннотации для всех изображений в папке."""
    face_detector = load_face_detector()
    if face_detector is None:
        print("Не удалось загрузить детектор лиц. Прерываем.")
        return
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"Обработка изображения: {image_path}")

            # Чтение изображения
            image = cv2.imread(image_path)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                continue

            # Детекция лица
            face_box, face_confidence = detect_face(image, face_detector)

            # Вычисляем атрибуты
            portrait_replace = compute_portrait_replace(image, face_box, face_confidence)
            printed_copy = compute_printed_copy(image)
            screen_reply = compute_screen_reply(image)

            # Сохраняем аннотацию
            save_annotation(image_path, portrait_replace, printed_copy, screen_reply)

if __name__ == "__main__":
    # Укажите пути к папкам с изображениями
    folders = [
        # "/Users/macbookair/PycharmProjects/idv3/data/train/original",
        "/Users/macbookair/PycharmProjects/idv3/data/train/fake",
        # "/Users/macbookair/PycharmProjects/idv3/data/val/original",
        "/Users/macbookair/PycharmProjects/idv3/data/val/fake"
    ]
    FACE_PROTO = "/Users/macbookair/PycharmProjects/idv3/opencv_face_detector.pbtxt"
    FACE_MODEL = "/Users/macbookair/PycharmProjects/idv3/opencv_face_detector_uint8.pb"
    for folder in folders:
        print(f"\nГенерация аннотаций для {folder}")
        generate_annotations_for_folder(folder)