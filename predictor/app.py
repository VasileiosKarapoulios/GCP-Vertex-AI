# predictor/app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from google.cloud import storage

AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI")
MODEL_FILENAME = "lora-llama"
LOCAL_TMP_DIR = "/tmp"
MODEL_ARTIFACT = None


def download_gcs_directory(gcs_uri, local_dir):
    """Downloads all files from a GCS URI prefix to a local directory."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS path.")

    # Remove 'gs://' prefix
    path_without_prefix = gcs_uri[5:]
    bucket_name, blob_prefix = path_without_prefix.split("/", 1)
    blob_prefix = blob_prefix.strip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # The prefix represents the "directory" in GCS
    blobs = bucket.list_blobs(prefix=blob_prefix)

    # Define the final local path where the model will be loaded from
    final_local_path = os.path.join(local_dir, os.path.basename(blob_prefix))
    os.makedirs(final_local_path, exist_ok=True)

    download_count = 0
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, blob_prefix)
        if relative_path == "." or relative_path.startswith(".."):
            continue

        local_file_path = os.path.join(final_local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        blob.download_to_filename(local_file_path)
        download_count += 1

    if download_count == 0:
        raise FileNotFoundError(f"No files found at GCS URI: {gcs_uri}")

    print(f"{download_count} files successfully downloaded to {final_local_path}")
    return final_local_path


print("Starting synchronous model initialization.")

if not AIP_STORAGE_URI:
    print("AIP_STORAGE_URI is not set. Cannot locate model artifacts in GCS.")
    raise EnvironmentError("AIP_STORAGE_URI is mandatory for custom model loading.")

try:
    local_model_path = download_gcs_directory(AIP_STORAGE_URI, LOCAL_TMP_DIR)

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    MODEL_ARTIFACT = AutoModelForCausalLM.from_pretrained(local_model_path)

    print("Model loading complete and successful.")

except Exception as e:
    raise RuntimeError(f"Model initialization failed: {e}")

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint required by Vertex AI."""
    if MODEL_ARTIFACT is not None:
        return jsonify({"status": "ready", "model_loaded": True}), 200
    else:
        return jsonify({"status": "error", "model_loaded": False}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction requests."""
    if not MODEL_ARTIFACT:
        return (
            jsonify({"error": "Model not loaded."}),
            503,
        )

    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = MODEL_ARTIFACT.generate(**inputs, max_new_tokens=100)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"predictions": [{"response": text}]})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
