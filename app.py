import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ✅ IMPORTANT: no pooling, because your model expects 7*7*1280 = 62720 features
enet_extractor = EfficientNetB0(weights="imagenet", include_top=False)

MODEL_PATH = os.path.join(app.root_path, "final_skin_model.keras")
classifier = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Light Diseases and Disorders of Pigmentation",
    "Acne and Rosacea Photos",
    "Systemic Disease",
    "Vascular Tumors",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Tinea Ringworm · Candidiasis · Fungal",
    "Psoriasis · Lichen Planus · Related",
    "Melanoma · Nevi · Moles",
    "Scabies · Lyme · Infestations",
    "Eczema Photos",
    "Seborrheic Keratoses · Benign Tumors",
    "Actinic Keratosis · BCC · Malignant",
    "Vasculitis Photos",
    "Cellulitis · Impetigo · Bacterial"
]

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded (field name must be 'image')"}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty file uploaded"}), 400

        # Read image
        try:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        # Preprocess
        img = img.resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        with tf.device("/CPU:0"):
            # Extract conv features: (1, 7, 7, 1280)
            features = enet_extractor.predict(x, verbose=0)

            # Flatten to (1, 62720) to match your classifier
            features = features.reshape((features.shape[0], -1))

            probs = classifier.predict(features, verbose=0)[0]

        if len(probs) != len(CLASS_NAMES):
            return jsonify({"error": f"Model output size ({len(probs)}) != CLASS_NAMES ({len(CLASS_NAMES)})."}), 500

        idx = int(np.argmax(probs))
        prediction = CLASS_NAMES[idx]
        confidence = round(float(np.max(probs)) * 100, 2)

        return jsonify({"prediction": prediction, "confidence": confidence}), 200

    except Exception as e:
        return jsonify({"error": f"Backend error: {str(e)}"}), 500

if __name__ == "__main__":
    print("✅ Model path:", MODEL_PATH)
    print("✅ Classifier input shape:", classifier.input_shape)  # should show (None, 62720)
    print("✅ Output units:", classifier.output_shape[-1])       # should show 15
    print("✅ Serving at http://127.0.0.1:5055")

    app.run(host="127.0.0.1", port=5055, debug=False, use_reloader=False, threaded=False)
