# trainer/train.py
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from google.cloud import storage
from google.cloud import aiplatform
import os
import torch
from tqdm import tqdm

from evaluate import load

token = ""  # HF token
BUCKET_NAME = "lora-bucket-474712"
DESTINATION_PATH = "model_artifacts/lora-llama"
EXPERIMENT_NAME = "lora-llama-experiment-final"
PROJECT_ID = "my-gcp-project-474712"
REGION = "europe-west1"
BUCKET_NAME = "lora-bucket-474712"

logging_enabled = False
try:
    # Initialize Vertex AI SDK.
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
    logging_enabled = True
    print(f"Vertex AI SDK initialized.")
    run_name = os.environ.get("AIP_TRAINING_JOB_ID", "lora-llama-run")
    aiplatform.start_run(run_name)
    print(f"Initialized Vertex AI Run: {run_name}")

except Exception as e:
    print(
        f"Failed to initialize Vertex AI SDK for logging. Metrics will not be logged. Error: {e}"
    )

# Load dataset
dataset_split = "train[:1%]"
dataset = load_dataset("tatsu-lab/alpaca", split=dataset_split)
# Keep only 4 samples to speed up (since it runs on CPU)
dataset = dataset.select(range(4))
dataset_length = len(dataset)

# Load model & tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_4bit=True,
    device_map="cpu",
    use_auth_token=token,
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


def format_prompt(example):
    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"


dataset = dataset.map(lambda x: {"text": format_prompt(x)})


def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


tokenized_ds = dataset.map(tokenize_fn, batched=True)
tokenized_ds = tokenized_ds.map(
    lambda samples: {"labels": samples["input_ids"]}, batched=True
)

# Training arguments
output_dir = "/tmp/lora-llama"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=False,
    save_strategy="epoch",
)

if logging_enabled:
    aiplatform.log_params(
        {
            # Training Arguments
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "fp16": training_args.fp16,
            # LoRA Configuration
            "base_model": model_name,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_target_modules": str(lora_config.target_modules),
            # Data Configuration
            "dataset_source": "tatsu-lab/alpaca",
            "dataset_split": dataset_split,
            "train_sample_count": dataset_length,
        }
    )
    print("Logged training parameters to Vertex AI.")


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
)
trainer.train()

# Save model in GCP storage
print("Merging LoRA adapter into base model...")
# Produces full model
model = model.merge_and_unload()

local_path = output_dir
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)
print(f"Full model saved at {local_path}")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

for root, dirs, files in os.walk(local_path):
    for file in files:
        local_file = os.path.join(root, file)
        relative_path = os.path.relpath(local_file, local_path)
        blob = bucket.blob(f"{DESTINATION_PATH}/{relative_path}")
        blob.upload_from_filename(local_file)

print(f"LoRA full model uploaded to gs://{BUCKET_NAME}/{DESTINATION_PATH}")


print("-" * 30)
print("Starting post-training evaluation...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
test_dataset = load_dataset("tatsu-lab/alpaca", split="train[1:2%]")
# Keep only 3 samples to speed up (since it runs on CPU)
test_dataset = test_dataset.select(range(3))
print(f"Loaded {len(test_dataset)} examples for evaluation.")

test_dataset = test_dataset.map(lambda x: {"text": format_prompt(x)})

predictions = []
references = []

for example in tqdm(test_dataset, desc="Generating Model Responses"):
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
    bertscore = load("bertscore")
    results_bertscore = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    avg_bertscore = sum(results_bertscore["f1"]) / len(results_bertscore["f1"])
    print(f"\nAverage BERTScore F1: {avg_bertscore:.4f}")
except Exception as e:
    print(f"BERTScore failed: {e}")
    avg_bertscore = None

if logging_enabled:
    evaluation_metrics = {
        "test_bertscore_f1": avg_bertscore,
    }
    aiplatform.log_metrics(evaluation_metrics)
    print(f"Logged final evaluation metrics to Vertex AI: {evaluation_metrics}")

print("\nEvaluation complete!")
aiplatform.end_run()
print("Finished Vertex AI Run.")
