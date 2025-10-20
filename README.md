### Pipeline to finetune (LoRA/Instruction Tuning), register and deploy an LLM with Vertex AI

## Essential components
1. Use Terraform to set up GCP Bucket, Artifact Registry and Endpoint.
2. Build Training and Serving Docker Images and push them to Artifact Registry.
3. Create a GCP Function that triggers when a new model is pushed to Model Registry and runs an evaluation script.
4. Create and Submit a Custom Training Job to GCP for instruction tuning of a small Llama model on the alpaca dataset.  
  4.5 Initiates also an experiment run in order to log metrics, parameters and useful information in Vertex AI Experiments.
5. Register the trained model to Model Registry.  
  5.5 Here trigger the function and runs the evaluation script on the latest model pushed to Model Registry.
6. Deploy the model to Endpoint.
7. Request a response from the model deployed to Endpoint through test.py.
