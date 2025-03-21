# ML Deployment Production 

ML Deployment Production is an end-to-end machine learning operations (MLOps) solution that demonstrates how to deploy machine learning models into production using tools like Databricks, MLflow, and GitHub Actions. This repository includes pipelines for feature engineering, model training, evaluation, and automated deployment.

## End to End Architecture
![databricks](https://github.com/user-attachments/assets/3e00251e-6497-49d2-b385-81eba3f4df4d)

## Overview

This repository implements a MLOps pipeline with the following components:

- **Feature Engineering:** Leverages Apache Spark on Databricks to process data and prepare features & labels. tables that are stored as delta tables.
- **Model Training & Evaluation:** Uses scikit-learn to build classification models, with experiment tracking and model registration handled by MLflow.
- **Automated Deployment:** Integrates with GitHub Actions to provide continuous integration and deployment (CI/CD) workflows and Jobs are scheduled for production environments.
- **Flexible Configuration:** Pipeline configurations are defined via YAML files and can be adjusted for development, staging, and production environments.

## Pipelines

The following pipelines currently defined within the package are:

_feature-table-creation_: Creates new feature table and separate labels Delta table in respective environments

_model-train_: Trains a scikit-learn Random Forest model

_model-inference-batch_: Load a model from MLflow Model Registry, load features from Feature Store, and score batch.

## Workflow
The CI/CD workflow is designed based on the end-to-end architecture of the MLOps pipeline:

#### Development & Integration:

- **Pull Requests & Testing:**
Developers create pull requests (PRs) for changes. Automated unit and integration tests run via GitHub Actions on every PR, ensuring code quality.

- **Branch Merging:** PRs are merged into the main branch once tests pass, triggering deployment workflows.

- **Automated Deployment:**
  - Scheduled Jobs: Production jobs are automatically scheduled. 
  - GitHub Actions Workflows: The repository’s .github/workflows/ directory contains the workflow definitions. These workflows:
    -   Run tests on code changes.
    -   Deploy and schedule Databricks jobs using the dbx tool.
    -   Manage environment transitions (dev → staging → prod).
    
- **Production Environment:**
    - Cluster Management: The jobs use an existing Databricks cluster (specified via existing_cluster_id). When a scheduled job is triggered, the cluster will turn on if it is not already running.
    - MLflow Tracking & Model Registry: Each model training run logs experiments and registers models using MLflow. This ensures traceability and enables model promotion based on evaluation metrics.


This workflow ensures that every code change is tested, integrated, and deployed in a structured manner, aligning with the repository's overall architecture for end-to-end ML model deployment.











