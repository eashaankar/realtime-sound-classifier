import os
import subprocess
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------------------
# Constants & configuration
# ----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Class names must match your training (and the order of the model's output layer)
CLASS_NAMES = ['baby_cry', 'boiling_water', 'door_knock', 'cat_meow', 'background', 'dog_bark']
NUM_CLASSES = len(CLASS_NAMES)
EMBEDDING_SIZE = 128

# VGGish model from TensorFlow Hub
VGGISH_MODEL_URL = 'https://tfhub.dev/google/vggish/1'

# VGGish expects 16 kHz mono audio
SAMPLE_RATE = 16000

# ----------------------------------------
# Flask app
# ----------------------------------------

app = Flask(__name__)
CORS(app)  # enable CORS for local dev

# ----------------------------------------
# Load models once at startup
# ----------------------------------------

print("Loading VGGish from TensorFlow Hub...")
GLOBAL_VGGISH_MODEL = hub.load(VGGISH_MODEL_URL)
print("VGGish loaded.")

MODEL_PATH = os.path.join(BASE_DIR, "final_audio_model.keras")
print(f"Loading classifier model from {MODEL_PATH}...")
classifier_model = tf.keras.models.load_model(MODEL_PATH)
print("Classifier model loaded.")

# ----------------------------------------
# Helpers
# ----------------------------------------

def convert_webm_to_wav(webm_path: str, wav_path: str, sr: int = SAMPLE_RATE) -> bool:
    """
    Use ffmpeg to convert WebM/Opus to WAV (mono, sr Hz).
    Returns True on success, False on failure.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", webm_path,
            "-ac", "1",
            "-ar", str(sr),
            wav_path,
        ]
        print(f"[INFO] Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("[ERROR] ffmpeg failed:")
            print(result.stderr)
            return False

        print(f"[INFO] ffmpeg conversion successful: {wav_path}")
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Make sure it is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"[ERROR] Exception during ffmpeg conversion: {e}")
        return False


def load_and_extract_feature(file_path: str):
    """
    Load audio file (WAV), resample to 16kHz, and compute a mean VGGish embedding.
    Returns a (128,) numpy array or None on error.

    IMPORTANT: VGGish TF Hub expects a 1-D waveform tensor of shape (None,),
    NOT a batched (1, N) tensor. So we pass the 1-D waveform directly.
    """
    try:
        # 1. Load audio at 16kHz mono
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        # 2. Convert to tf.Tensor of shape (num_samples,)
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        # 3. Get embeddings from VGGish
        embeddings = GLOBAL_VGGISH_MODEL(waveform).numpy()
        print(f"[INFO] VGGish embeddings shape: {embeddings.shape}")

        # Typical shape is (T, 128). If there's an extra batch dim, squeeze it.
        if embeddings.ndim == 3:
            embeddings = np.squeeze(embeddings, axis=0)  # shape: (T, 128)

        # 4. Average over time frames to get single vector
        if embeddings.shape[0] > 0:
            mean_embedding = np.mean(embeddings, axis=0)  # shape: (128,)
            return mean_embedding
        else:
            print(f"[WARN] No embeddings generated for {file_path}, using zeros.")
            return np.zeros((EMBEDDING_SIZE,), dtype=np.float32)

    except Exception as e:
        print(f"[ERROR] Error processing {file_path}: {e}")
        return None

# ----------------------------------------
# Routes
# ----------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_audio():
    temp_webm_path = None
    temp_wav_path = None

    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio_file part in the request"}), 400

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save uploaded file temporarily (browser sends audio/webm)
        temp_webm_path = os.path.join(BASE_DIR, "temp_audio.webm")
        temp_wav_path = os.path.join(BASE_DIR, "temp_audio.wav")

        audio_file.save(temp_webm_path)
        print(f"[INFO] Saved uploaded file to {temp_webm_path}")

        # Convert WebM -> WAV using ffmpeg
        ok = convert_webm_to_wav(temp_webm_path, temp_wav_path, sr=SAMPLE_RATE)
        if not ok:
            return jsonify({
                "error": "Failed to convert audio with ffmpeg. "
                         "Ensure ffmpeg is installed and accessible from PATH."
            }), 500

        # Extract features from the WAV file
        features = load_and_extract_feature(temp_wav_path)
        if features is None:
            return jsonify({
                "error": "Feature extraction failed even after ffmpeg conversion."
            }), 500

        # Model expects shape (batch_size, 128)
        features = np.expand_dims(features, axis=0)

        # Run prediction
        predictions = classifier_model.predict(features)
        probs = predictions[0]
        predicted_class_index = int(np.argmax(probs))
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Map each class name to its probability (nice for frontend)
        prob_dict = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "predicted_class": predicted_class_name,
            "probabilities": probs.tolist(),      # list, index-aligned with CLASS_NAMES
            "probabilities_dict": prob_dict       # dict, label -> probability
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500
    finally:
        # Clean up temp files
        for p in (temp_webm_path, temp_wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

# ----------------------------------------
# Entry point
# ----------------------------------------

if __name__ == "__main__":
    # For local development / testing
    app.run(host="0.0.0.0", port=5001, debug=True)
