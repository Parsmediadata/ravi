# myapp.py
from flask import Flask, request, jsonify
import os
import cv2
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# آماده‌سازی مدل
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "هیچ فایلی ارسال نشده"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "نام فایل خالی است"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # بارگذاری و پردازش تصویر
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "فایل تصویر معتبر نیست"}), 400

    faces = face_app.get(image)
    if not faces:
        return jsonify({"error": "هیچ چهره‌ای شناسایی نشد"}), 400

    embedding = faces[0].embedding.tolist()
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

