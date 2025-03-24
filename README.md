# ML Deployment Production 

ML Deployment Production is an end-to-end MLOps solution that demonstrates how to seamlessly deploy machine learning models into production using Databricks, Delta Lake, MLflow, and GitHub Actions. This repository includes pipelines for feature engineering, model training, evaluation, and automated deployment, integrating DataOps, ModelOps, and DevOps into a single, robust workflow.

Leverage a comprehensive Airbnb dataset to build a powerful classifier that predicts whether a booking will be canceled. By proactively identifying high-risk reservations, you can optimize inventory management, enhance customer satisfaction, and streamline operations. If you don’t already have the dataset at hand, simply upload the Booking.csv file to your data lake to jumpstart data ingestion and get your predictive pipeline up and running.

**Security considerations:**
- GitHub: Fine-Grained Access Tocken: Repository is added to an FG token.
- Databricks: Admin token behind the environment
- Databricks jobs will be automated using Service Principal
- Databricks permissions: User Based on workspace, model, experiment, and model.

### Deployment Strategy

![image](https://github.com/user-attachments/assets/c80c14d7-1f7b-4ade-8d43-17d70d74323f)

The deployment strategy following in this project is the Deploy Code pattern, ie Code for an ML project is developed in the development environment, and this code is then moved to the staging environment, where it is tested. Following successful testing, the project code is deployed to the production environment, where it is executed. Model training code is tested in the staging environment using a subset of data, and the model training pipeline is executed in the production environment. The model deployment process of validating a model and additionally conducting comparisons versus any existing production model all run within the production environment. (Here we will be having a dev/stage/prod environment in a single workspace separated by a folder hierarchy)

## End to End Architecture
![databricks](https://github.com/user-attachments/assets/3e00251e-6497-49d2-b385-81eba3f4df4d)

## Overview

This repository implements a MLOps pipeline with the following components:

- **Feature Engineering:** Leverages Apache Spark on Databricks to process data and prepare features & labels. tables that are stored as delta tables.
- **Model Training & Evaluation:** Uses sci-kit-learn to build classification models, with experiment tracking and model registration handled by MLflow.
- **Automated Deployment:** Integrates with GitHub Actions to provide continuous integration and deployment (CI/CD) workflows and Jobs are scheduled for production environments.
- **Flexible Configuration:** All pipeline configurations are defined in YAML files, enabling the same codebase to adapt seamlessly across development, staging, and production environments. By adjusting the YAML parameters, it becomes straightforward to tailor table names, model storage paths, pipeline inputs, and other job-specific settings without altering core code.
      -  Each environment reads parameters from its corresponding YAML file, ensuring consistency and preventing drift.  The conf/deployment.yml file defines how and when jobs are executed, including scheduling and dependencies among tasks.
      -  This approach unifies multiple pipelines—such as data ingestion, feature engineering, model training, and inference—under a single deployment configuration. By specifying cron schedules or other triggers, jobs can run on a set cadence (e.g., daily, hourly) without manual intervention.


## Pipelines

The following pipelines currently defined within the package and will be deployed are:

- _feature-table-creation_: Creates new feature table and separate labels Delta table in respective environments

- _model-train: Harness the power of a scikit-learn Random Forest to automatically train and register the trained model at MLFlow Model Registry. If a previous version exists, the pipeline seamlessly compares performance and promotes only the best model to Production—ensuring your deployed model is always the top performer.

- _model-inference-batch_: Load a model from MLflow Model Registry, load features from the Feature table at Delta Lake, score batch, and write the results back in Delta Lake.

## Workflow
The CI/CD workflow is designed based on the end-to-end architecture of the MLOps pipeline:

#### Development & Integration:

- **Pull Requests & Testing:**
Developers create pull requests (PRs) for changes. Automated unit and integration tests run via GitHub Actions on every PR, ensuring code quality during stage env. Here I have also replicated the entire production scenario in the stage to ensure the checkboxes of integration.

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

#### Deployed production Deliverables:
      - Production Databricks Jobs: https://adb-4004511372084821.1.azuredatabricks.net/jobs/225946478137713?o=4004511372084821
      - Experiments: https://adb-4004511372084821.1.azuredatabricks.net/ml/experiments/3378579225337397/runs/f0eaa454b66249b6a7869f2c3a444ac9?o=4004511372084821
      - Model Registry: https://adb-4004511372084821.1.azuredatabricks.net/ml/models/booking_prediction_production?o=4004511372084821
