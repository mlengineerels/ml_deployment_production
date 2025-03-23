# ML Deployment Production 

ML Deployment Production is an end-to-end machine learning operations (MLOps) solution that demonstrates how to deploy machine learning models into production using tools like Databricks, MLflow, and GitHub Actions. This repository includes pipelines for feature engineering, model training, evaluation, and automated deployment.

The use case at hand is a booking cancellation prediction problem. Here I will be using Airbnb dataset to build a simple classifier to predict whether a booking will be canclelled from a existing bookings. If the data set is not presnt with you please feel free to upload the mentioned Booking.csv file in the datalake or run the setup script inorder to kickstart the process of ingesting the data.

### Deployment Strategy

![image](https://github.com/user-attachments/assets/c80c14d7-1f7b-4ade-8d43-17d70d74323f)

The deployment strategy following in this project is Deploy Code pattern, ie Code for an ML project is developed in the development environment, and this code is then moved to the staging environment, where it is tested. Following successful testing, the project code is deployed to the production environment, where it is executed.Model training code is tested in the staging environment using a subset of data, and the model training pipeline is executed in the production environment.The model deployment process of validating a model and additionally conducting comparisons versus any existing production model all run within the production environment.





**Security considerations:**
- GitHub: Fine Grained Access Tocken: Repository is added to a FG token.
- Databricks: Admin tocken behind the environment
- Databricks jobs will be automted using Service Principal
- Databricks permissions: User Based for workspace, model, experiment and model.

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

- _feature-table-creation_: Creates new feature table and separate labels Delta table in respective environments

- model-train: Harness the power of a scikit-learn Random Forest to automatically train and register the trained model at MLFlow Model ZRegistry. If a previous version exists, the pipeline seamlessly compares performance and promotes only the best model to Production—ensuring your deployed model is always the top performer.

- _model-inference-batch_: Load a model from MLflow Model Registry, load features from Feature table at delta lake, score batch and write the results back in delta lake.

## Workflow
The CI/CD workflow is designed based on the end-to-end architecture of the MLOps pipeline:

#### Development & Integration:

- **Pull Requests & Testing:**
Developers create pull requests (PRs) for changes. Automated unit and integration tests run via GitHub Actions on every PR, ensuring code quality during stage env. In here I have also replicated entire production scenario in the stage to ensure the checkboxes of integration.

- **Branch Merging:** PRs are merged into the main branch once tests pass, triggering deployment workflows.

- **Automated Deployment:**
  - Scheduled Jobs: Production jobs are automatically scheduled. 
  - GitHub Actions Workflows: The repository’s .github/workflows/ directory contains the workflow definitions. These workflows:
    -   Run tests on code changes.
    -   Deploy and schedule Databricks jobs using the dbx tool.
    -   Manage environment transitions (dev → staging → prod).
    
- **Production Environment:**
    - You can configure the job to run on an existing Databricks cluster (specified by existing_cluster_id), which will automatically start if it’s not already running. Alternatively, you can create an ephemeral job cluster that spins up exclusively for the job and is deleted once it completes—ensuring efficient resource usage and cost control.
    - MLflow Tracking & Model Registry: Each model training run logs experiments and registers models using MLflow. This ensures traceability and enables model promotion based on evaluation metrics.

This workflow ensures that every code change is tested, integrated, and deployed in a structured manner, aligning with the repository's overall architecture for end-to-end ML model deployment.














