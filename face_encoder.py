import base64
import io
import json
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import face_recognition

app = FastAPI()

class ImageInput(BaseModel):
    image_base64: str

class EncodingOutput(BaseModel):
    encodings: List[List[float]]

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="فرمت تصویر نامعتبر است.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطا در تبدیل base64 به تصویر: {str(e)}")

@app.post("/encode_face", response_model=EncodingOutput)
def encode_face(image_input: ImageInput):
    try:
        img_array = decode_base64_image(image_input.image_base64)
        face_locations = face_recognition.face_locations(img_array)

        if not face_locations:
            raise HTTPException(status_code=404, detail="No face found in the image.")

        encodings = face_recognition.face_encodings(img_array, face_locations)

        return {"encodings": [enc.tolist() for enc in encodings]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
