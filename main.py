import base64
import io
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import numpy as np
import face_recognition

app = FastAPI()

class ImageInput(BaseModel):
    image_base64: str

class EncodingOutput(BaseModel):
    encodings: List[List[float]]

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # تبدیل به RGB برای جلوگیری از خطا
        return np.array(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="فرمت تصویر نامعتبر است.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطا در تبدیل base64 به تصویر: {str(e)}")

@app.post("/encode_face", response_model=EncodingOutput)
def encode_face(image_input: ImageInput):
    try:
        print("📥 Base64 دریافت شد.")
        img_array = decode_base64_image(image_input.image_base64)
        print("✅ تصویر با موفقیت decode شد.")

        face_locations = face_recognition.face_locations(img_array)
        print(f"🔍 تعداد چهره‌های شناسایی‌شده: {len(face_locations)}")

        if not face_locations:
            raise HTTPException(status_code=404, detail="هیچ چهره‌ای در تصویر یافت نشد.")

        encodings = face_recognition.face_encodings(img_array, face_locations)

        print("✅ چهره‌ها encode شدند.")
        return {"encodings": [enc.tolist() for enc in encodings]}

    except HTTPException:
        raise  # ارورهایی که خودمان تعریف کردیم بدون تغییر دوباره ارسال می‌شود
    except Exception as e:
        print(f"❌ خطای کلی: {str(e)}")
        raise HTTPException(status_code=500, detail="خطای داخلی سرور.")
