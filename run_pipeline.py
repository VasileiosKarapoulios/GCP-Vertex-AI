# run_pipeline.py
from google.cloud import aiplatform

PROJECT_ID = "my-gcp-project-474712"
REGION = "europe-west1"
BUCKET_NAME = "lora-bucket-474712"
JOB_DISPLAY_NAME = "lora-training-job"
MODEL_DISPLAY_NAME = "lora-llama-model"
AR_REPO_NAME = "lora-ar-474712"  # Artifact Registry
TRAIN_IMAGE_NAME = "lora-train-img"
SERVE_IMAGE_NAME = "lora-serve-img"
MODEL_ARTIFACT_URI = f"gs://{BUCKET_NAME}/model_artifacts/lora-llama"
ENDPOINT_ID = "1010101010"
MACHINE_TYPE = "n1-standard-8"

AR_TRAIN_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_NAME}/{TRAIN_IMAGE_NAME}:latest"
)
AR_SERVE_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_NAME}/{SERVE_IMAGE_NAME}:latest"
)


aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

# Custom training job to finetune the model
print(f"Creating Custom Training Job: {JOB_DISPLAY_NAME}")
job = aiplatform.CustomContainerTrainingJob(
    display_name=JOB_DISPLAY_NAME,
    container_uri=AR_TRAIN_IMAGE_URI,
)

print("Submitting Custom Training Job...")
model = job.run(
    machine_type=MACHINE_TYPE,
    replica_count=1,
    args=[f"--model-dir={MODEL_ARTIFACT_URI}"],
    sync=True,
    service_account="785748189449-compute@developer.gserviceaccount.com",
)
print(f"Training job finished. Model artifacts saved to: {MODEL_ARTIFACT_URI}")


print(f"Registering Model: {MODEL_DISPLAY_NAME}")

existing_models = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')

parent_model_resource_name = None
is_new_model = True

if existing_models:
    parent_model_resource_name = existing_models[0].resource_name
    is_new_model = False
    print(
        f"Parent model found ({parent_model_resource_name}). Registering as a new version."
    )
else:
    print("Parent model not found. Registering as a new model (Version 1).")

# Register model to Model Registry
uploaded_model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=MODEL_ARTIFACT_URI,
    serving_container_image_uri=AR_SERVE_IMAGE_URI,
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_ports=[8080],
    parent_model=parent_model_resource_name,
    is_default_version=True,
    sync=True,
)
print(f"Model registered: {uploaded_model.resource_name}")

endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
print(f"Deploying Model to Endpoint: {endpoint}")

# Deploy model to Endpoint
endpoint.deploy(
    model=uploaded_model,
    deployed_model_display_name=MODEL_DISPLAY_NAME,
    machine_type=MACHINE_TYPE,
    min_replica_count=1,
    max_replica_count=1,
    sync=True,
)
print(f"Model deployed to endpoint: {endpoint.resource_name}")
print(f"Endpoint URL: {endpoint.display_name}")
