# build_and_push_images.py
import os
from google.cloud import aiplatform

PROJECT_ID = "my-gcp-project-474712"
REGION = "europe-west1"
BUCKET_NAME = "lora-bucket-474712"
AR_REPO_NAME = "lora-ar-474712"
TRAIN_IMAGE_NAME = "lora-train-img"
SERVE_IMAGE_NAME = "lora-serve-img"

AR_TRAIN_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_NAME}/{TRAIN_IMAGE_NAME}:latest"
)
AR_SERVE_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_NAME}/{SERVE_IMAGE_NAME}:latest"
)

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)


def build_and_push_image(dockerfile, image_uri):
    """Builds the Docker image and pushes it to Artifact Registry."""
    print(f"Building and pushing {dockerfile} to {image_uri}...")

    command = f"gcloud builds submit {dockerfile} " f"--tag {image_uri}"

    if os.system(command) != 0:
        raise RuntimeError(f"Failed to build and push {dockerfile}")
    print("Build and push successful.")


# Build and push training container
build_and_push_image("trainer/", AR_TRAIN_IMAGE_URI)

# Build and push serving container
build_and_push_image("predictor/", AR_SERVE_IMAGE_URI)
