from evaluate import load
import bert_score
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import base64
import os
from google.cloud import storage

from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import aiplatform
from google.api_core.exceptions import NotFound

PROJECT_ID = "my-gcp-project-474712"
REGION = "europe-west1"
MODEL_DISPLAY_NAME = "lora-llama-model"


def download_gcs_folder_to_local_tmp(gcs_uri, local_dir):
    """
    Downloads a folder from GCS (gs://bucket/path) to a local directory (/tmp/local_dir).

    Args:
        gcs_uri (str): The GCS path (e.g., 'gs://bucket-name/folder-path').
        local_dir (str): The local path where the folder contents should be saved.
    """
    try:
        # Extract bucket name and folder prefix
        if not gcs_uri.startswith("gs://"):
            raise ValueError("GCS URI must start with 'gs://'")

        parts = gcs_uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError("GCS URI format must be gs://bucket/path/to/folder")

        bucket_name, prefix = parts
        prefix = prefix.rstrip("/") + "/"

        # Initialize GCS client and create local directory
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading GCS contents from {gcs_uri} to local path {local_dir}...")

        blobs = bucket.list_blobs(prefix=prefix)

        found_files = 0
        for blob in blobs:
            if blob.name == prefix:
                continue

            relative_path = blob.name[len(prefix) :]
            local_file = os.path.join(local_dir, relative_path)

            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            blob.download_to_filename(local_file)
            found_files += 1

        if found_files == 0:
            print(f"Warning: No files found in GCS path {gcs_uri}.")

        print(f"Successfully downloaded {found_files} files to {local_dir}.")

    except Exception as e:
        print(f"GCS Download Error: {e}")
        raise

    return local_dir


def format_prompt(example):
    """Formats the Alpaca instruction/input/output into a single prompt string."""
    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"


def evaluate_model(project_id, region, model_display_name):
    """
    Loads the latest model version from Vertex AI Model Registry and evaluates it.

    Args:
        project_id: GCP ID
        region: GC Region
        model_display_name: The display name of the model in the registry
    """
    print(f"Starting evaluation for model: {model_display_name} in region: {region}")

    model_gcs_uri = None
    full_model_version = None

    try:
        aiplatform.init(project=project_id, location=region)

        # List all Parent Models matching the display name
        all_models = aiplatform.Model.list(location=region)
        models = [m for m in all_models if m.display_name == model_display_name]

        if not models:
            raise NotFound(
                f"Parent Model with display name '{model_display_name}' not found in Model Registry."
            )

        parent_model = models[0]

        #  List all versions
        versions = parent_model.versioning_registry.list_versions()

        if not versions:
            raise NotFound(f"No versions found for model '{model_display_name}'")

        # Sort versions by ID
        latest_version_info = sorted(
            versions, key=lambda v: int(v.version_id), reverse=True
        )[0]

        model_version_resource_name = (
            f"{parent_model.resource_name}@{latest_version_info.version_id}"
        )

        full_model_version = aiplatform.Model(model_version_resource_name)

        model_gcs_uri = full_model_version.uri
        model_version_id = latest_version_info.version_id

        print(f"Found latest model version: {model_version_id}")
        print(f"Artifact URI for latest version: {model_gcs_uri}")

    except NotFound as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error retrieving model from Vertex AI: {e}")
        return

    LOCAL_MODEL_PATH = "/tmp/local_model"
    model_load_uri = ""
    try:
        # Download files from GCS to the local /tmp directory
        model_load_uri = download_gcs_folder_to_local_tmp(
            model_gcs_uri, LOCAL_MODEL_PATH
        )
        print(f"Attempting to load model from local path: {model_load_uri}")
    except Exception as e:
        print(f"Failed to download model artifacts. Error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_load_uri)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_load_uri,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        ).to(device)
        model.eval()
        print("Model and tokenizer loaded successfully.")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise e

    dataset = load_dataset("tatsu-lab/alpaca", split="train[1:2%]")
    # Keep only 3 samples to speed up the process (since it runs on CPU)
    dataset = dataset.select(range(3))
    print(f"Loaded {len(dataset)} examples for evaluation.")

    dataset = dataset.map(lambda x: {"text": format_prompt(x)})

    predictions = []
    references = []

    for example in tqdm(dataset, desc="Generating Model Responses"):
        prompt = example["text"].split("### Response:")[0] + "\n### Response:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        model_response = (
            decoded.split("### Response:")[-1].strip()
            if "### Response:" in decoded
            else decoded.strip()
        )

        predictions.append(model_response)
        references.append(example["output"])

    try:
        (P, R, F1) = bert_score.score(predictions, references, lang="en", verbose=True)
        avg_bertscore = F1.mean().item()
        print(f"\nAverage BERTScore F1: {avg_bertscore:.4f}")
    except Exception as e:
        print(f"BERTScore failed: {e}")
        avg_bertscore = None

    print(
        "\nEvaluation complete! Results could now be pushed to a database or dashboard."
    )

    return {
        "model_name": full_model_version.display_name,
        "model_version": full_model_version.version_id,
        "metrics": {
            "bertscore_f1": avg_bertscore,
        },
    }


def start_evaluation(event, context):
    """
    Pub/Sub Cloud Function entry point.
    Triggered by a message from Cloud Audit Logs indicating a new Vertex AI Model upload.

    The function relies on the Sink Filter to ensure this is an 'UploadModel' event
    and uses the static MODEL_DISPLAY_NAME to fetch the latest version.
    """
    print(f"Cloud Function triggered by event ID: {context.event_id}")

    if not all([PROJECT_ID, REGION]):
        raise ValueError(
            "Missing required configuration: PROJECT_ID, REGION is not set."
        )

    try:
        if "data" in event:
            pubsub_message_data = base64.b64decode(event["data"]).decode("utf-8")
            print(
                f"Decoded Pub/Sub message data (first 500 chars): {pubsub_message_data[:500]}..."
            )

    except Exception as e:
        print(
            f"Warning: Could not decode Pub/Sub message data: {e}. Proceeding with evaluation based on configured model name."
        )

    evaluation_result = evaluate_model(
        project_id=PROJECT_ID,
        region=REGION,
        model_display_name=MODEL_DISPLAY_NAME,
    )

    print(f"Final Evaluation Result: {json.dumps(evaluation_result)}")
