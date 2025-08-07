import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load the main classification model
def load_model(model_path: str):
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Updated to use weights
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 5)  # 5 outputs: 2 for classification, 3 for attributes
    model = nn.Sequential(
        resnet,
        nn.Dropout(p=0.5)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Load the screen artifact detection model
screen_model = models.resnet18(weights=None)  # No pretrained weights, use trained model
screen_model.fc = torch.nn.Linear(screen_model.fc.in_features, 2)  # Binary classification
screen_model.load_state_dict(torch.load("models/screen_artifact_classifier.pth", map_location=device))
screen_model.eval()
screen_model.to(device)

# Image transform for screen model
screen_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Image transform for main model
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Инференс
def predict(model, image: Image.Image):
    # Preprocess for main model
    tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)  # Shape: [1, 5]
        class_logits = output[:, 0:2]  # First two for classification
        probs = F.softmax(class_logits, dim=1)
        original_prob = probs[0, 0].item()  # Probability for "original"
        fake_prob = probs[0, 1].item()  # Probability for "fake"

        # Extract attribute predictions (excluding screenReply for now)
        portrait_replace = torch.sigmoid(output[:, 2]).item()
        printed_copy = torch.sigmoid(output[:, 3]).item()
        # screenReply will be computed separately with the screen model

    # Preprocess for screen model
    screen_tensor = screen_transform(np.array(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        screen_output = screen_model(screen_tensor)
        screen_probs = F.softmax(screen_output, dim=1)
        screen_reply = screen_probs[0, 1].item()  # Probability of screen artifact (class 1)

    # Thresholds
    original_confidence_threshold = 0.7
    attribute_threshold = 0.5

    # Determine reason and label
    reasons = []
    if portrait_replace < attribute_threshold:
        reasons.append(f"Portrait replacement detected (confidence: {portrait_replace:.3f})")
    if printed_copy < attribute_threshold:
        reasons.append(f"Printed copy detected (confidence: {printed_copy:.3f})")
    if screen_reply < attribute_threshold:
        reasons.append(f"Screen replay detected (confidence: {screen_reply:.3f})")

    # New decision logic
    if original_prob > original_confidence_threshold and reasons:
        # Case 1: confidence_original > 0.7 and any attribute < 0.6
        label = "fake"
        reason = "; ".join(reasons)
    elif original_prob < original_confidence_threshold and not (
            portrait_replace < attribute_threshold or
            printed_copy < attribute_threshold or
            screen_reply < attribute_threshold
    ):
        # Case 2: confidence_original < 0.7 and all attributes > 0.6
        label = "original"
        reason = "No specific issues detected"
    else:
        # Default case: fake if any attribute < 0.6, else original
        label = "fake" if reasons else "original"
        reason = "; ".join(reasons) if reasons else "No specific issues detected"

    return {
        "label": label,
        "confidence_original": original_prob,
        "confidence_fake": fake_prob,
        "portraitReplace": portrait_replace,
        "printedCopy": printed_copy,
        "screenReply": screen_reply,
        "status": "Ok",
        "reason": reason
    }