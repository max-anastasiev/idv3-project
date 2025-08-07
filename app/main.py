from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import cv2
import numpy as np
import imghdr
from app.models_utils import predict, load_model, screen_model

app = FastAPI()
model = load_model("models/resnet18_id_classifier.pth")  # Load globally


@app.post("/detect_liveness", summary="Detect liveness of ID document with model classification")
async def detect_liveness(document_files: list[UploadFile] = File(...)):
    if not document_files:
        raise HTTPException(status_code=400, detail="At least one document image is required.")

    document_contents = [await file.read() for file in document_files]

    # Strict check for JPG/JPEG format
    for i, (contents, file) in enumerate(zip(document_contents, document_files)):
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ["jpg", "jpeg"]:
            raise HTTPException(status_code=400,
                                detail=f"Invalid file format for frame {i + 1}. Only .jpg or .jpeg formats are supported.")
        # Verify image content type
        img_type = imghdr.what(None, h=contents)
        if img_type not in ["jpeg", "jpg"]:
            raise HTTPException(status_code=400,
                                detail=f"Invalid image format for frame {i + 1}. Uploaded file is not a valid .jpg or .jpeg image.")

    document_nps = [cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR) for contents in document_contents]

    if any(doc is None for doc in document_nps):
        raise HTTPException(status_code=400, detail="Failed to decode one or more images")

    # Convert numpy arrays to PIL Images and predict
    results = []
    for np_image in document_nps:
        pil_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
        result = predict(model, pil_image)
        results.append(result)

    # Average the metrics across all images
    num_images = len(results)
    averaged_result = {
        "confidence_original": sum(r["confidence_original"] for r in results) / num_images,
        "confidence_fake": sum(r["confidence_fake"] for r in results) / num_images,
        "portraitReplace": sum(r["portraitReplace"] for r in results) / num_images,
        "printedCopy": sum(r["printedCopy"] for r in results) / num_images,
        "screenReply": sum(r["screenReply"] for r in results) / num_images,
    }

    # Determine label and reason based on averaged values
    original_confidence_threshold = 0.7
    attribute_threshold = 0.6

    reasons = []
    if averaged_result["portraitReplace"] < attribute_threshold:
        reasons.append(f"Portrait replacement detected (confidence: {averaged_result['portraitReplace']:.3f})")
    if averaged_result["printedCopy"] < attribute_threshold:
        reasons.append(f"Printed copy detected (confidence: {averaged_result['printedCopy']:.3f})")
    if averaged_result["screenReply"] < attribute_threshold:
        reasons.append(f"Screen replay detected (confidence: {averaged_result['screenReply']:.3f})")

    if averaged_result["confidence_original"] > original_confidence_threshold and reasons:
        label = "fake"
        reason = "; ".join(reasons)
    elif averaged_result["confidence_original"] < original_confidence_threshold and not (
            averaged_result["portraitReplace"] < attribute_threshold or
            averaged_result["printedCopy"] < attribute_threshold or
            averaged_result["screenReply"] < attribute_threshold
    ):
        label = "original"
        reason = "No specific issues detected"
    else:
        label = "fake" if reasons else "original"
        reason = "; ".join(reasons) if reasons else "No specific issues detected"

    final_result = {
        "label": label,
        "confidence_original": averaged_result["confidence_original"],
        "confidence_fake": averaged_result["confidence_fake"],
        "portraitReplace": averaged_result["portraitReplace"],
        "printedCopy": averaged_result["printedCopy"],
        "screenReply": averaged_result["screenReply"],
        "status": "Ok",
        "reason": reason
    }

    return {"result": final_result, "resultCode": "Ok"}