from google.cloud import aiplatform

PROJECT_ID = "my-gcp-project-474712"
REGION = "europe-west1"
ENDPOINT_ID = "1010101010"

aiplatform.init(project=PROJECT_ID, location=REGION)

endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

instances = [
    {
        "prompt": "### Instruction:\n What's the best football player in the world?\n### Response:\n"
    }
]
try:
    response = endpoint.predict(instances=instances)
    print(response.predictions[0]["response"])
except Exception as e:
    print(f"Prediction failed: {e}")
