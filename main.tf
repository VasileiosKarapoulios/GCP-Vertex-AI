variable "project_id" { default = "my-gcp-project-474712" }
variable "region" { default = "europe-west1" }
variable "bucket_name" { default = "lora-bucket-474712" }
variable "ar_repo_name" { default = "lora-ar-474712" }
variable "endpoint_display_name" { default = "lora-endpoint-474712" }
variable "model_display_name" { default = "lora-finetuned-474712" }

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
  ])
  project = var.project_id
  service = each.key
  disable_on_destroy = false
}

# Artifact Registry
resource "google_artifact_registry_repository" "ar_repo" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = var.ar_repo_name
  format        = "DOCKER"
}

# Cloud Storage Bucket
resource "google_storage_bucket" "lora_bucket" {
  project      = var.project_id
  name         = var.bucket_name
  location     = upper(var.region)
  uniform_bucket_level_access = true
}

# Vertex AI Endpoint
resource "google_vertex_ai_endpoint" "lora_endpoint" {
  project      = var.project_id
  location     = var.region
  name         = "1010101010"
  display_name = var.endpoint_display_name
}
