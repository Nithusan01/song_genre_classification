import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Config
MODEL_PATH = "models/best_model_no_val.h5"
UPLOAD_FOLDER = "uploads"
TEMP_SPEC_PATH = "temp_spec.png"
IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
CLASS_NAMES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at start
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def audio_to_spectrogram(filepath, save_path=TEMP_SPEC_PATH):
    y, sr = librosa.load(filepath, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(2.56, 2.56))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.margins(0,0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='inferno')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close()

def prepare_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part in request"
            return render_template("index.html", error=error)

        file = request.files['file']
        if file.filename == '':
            error = "No file selected"
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                audio_to_spectrogram(filepath)
                img = prepare_image(TEMP_SPEC_PATH)
                preds = model.predict(img)
                pred_idx = np.argmax(preds)
                prediction = CLASS_NAMES[pred_idx]
                confidence = float(preds[0][pred_idx])
            except Exception as e:
                error = f"Prediction error: {e}"
                print(error)

        else:
            error = "File type not allowed. Upload wav or mp3."

    return render_template("index.html", prediction=prediction, confidence=confidence, error=error)

if __name__ == "__main__":
    app.run(debug=True)
