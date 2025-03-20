from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pprint
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from booking.common import MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from booking.model_train_pipeline import ModelTrainPipeline
from booking.utils.logger_utils import get_logger

_logger = get_logger()

spark = SparkSession.builder.appName('modeltrain').getOrCreate()

@dataclass
class ModelTrainConfig:
    """
    Configuration data class used to execute the ModelTrain pipeline.

    Attributes:
        mlflow_tracking_cfg (MLflowTrackingConfig):
            Configuration for MLflow parameters during a model training run.
        feature_store_table_cfg (FeatureStoreTableConfig):
            Configuration used when reading the feature delta table.
        labels_table_cfg (LabelsTableConfig):
            Configuration used when loading the labels table.
        pipeline_params (dict):
            Parameters for the preprocessing pipeline. (e.g., test_size, random_state)
        model_params (dict):
            Dictionary of parameters for the model (read from model_train.yml).
        conf (dict):
            [Optional] Configuration file contents to log to MLflow.
        env_vars (dict):
            [Optional] Environment variables to log to MLflow.
    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    feature_store_table_cfg: FeatureStoreTableConfig
    labels_table_cfg: LabelsTableConfig
    pipeline_params: Dict[str, Any]
    model_params: Dict[str, Any]
    conf: Dict[str, Any] = None
    env_vars: Dict[str, str] = None

class ModelTrain:
    """
    Class to execute model training. Parameters, metrics, and model artifacts are tracked with MLflow.
    Optionally, the resulting model will be registered in the MLflow Model Registry.
    """
    def __init__(self, cfg: ModelTrainConfig):
        self.cfg = cfg

    @staticmethod
    def _set_experiment(mlflow_tracking_cfg: MLflowTrackingConfig):
        """
        Sets the MLflow experiment using either an experiment_id or an experiment_path.
        """
        if mlflow_tracking_cfg.experiment_id is not None:
            _logger.info(f'MLflow experiment_id: {mlflow_tracking_cfg.experiment_id}')
            mlflow.set_experiment(experiment_id=mlflow_tracking_cfg.experiment_id)
        elif mlflow_tracking_cfg.experiment_path is not None:
            _logger.info(f'MLflow experiment_path: {mlflow_tracking_cfg.experiment_path}')
            mlflow.set_experiment(experiment_name=mlflow_tracking_cfg.experiment_path)
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    def _get_feature_table_lookup(self) -> Tuple[DataFrame, List[str]]:
        """
        Reads the feature delta table using Spark and returns:
          - A Spark DataFrame of the feature table.
          - A list of lookup keys (primary keys) for joining with the labels table.
        """
        feature_store_table_cfg = self.cfg.feature_store_table_cfg

        _logger.info("Reading feature delta table using Spark...")
        table_name = f"{feature_store_table_cfg.database_name}.{feature_store_table_cfg.table_name}"
        df = spark.read.table(table_name)
        lookup_keys = feature_store_table_cfg.primary_keys

        return df, lookup_keys

    def get_training_data(self) -> DataFrame:
        """
        Reads the feature delta table and joins it with the labels table using Spark.
        
        Returns:
            A Spark DataFrame resulting from an inner join on the lookup keys.
        """
        labels_table_cfg = self.cfg.labels_table_cfg

        _logger.info("Reading feature delta table...")
        features_df, lookup_keys = self._get_feature_table_lookup()

        _logger.info("Reading labels table...")
        labels_df = spark.table(f'{labels_table_cfg.database_name}.{labels_table_cfg.table_name}')
        
        _logger.info(f"Joining labels with features on keys: {lookup_keys}")
        training_df = labels_df.join(features_df, on=lookup_keys, how='inner')
        return training_df

    def create_train_test_split(self, training_df: DataFrame):
        """
        Converts the Spark DataFrame to pandas and creates train/test splits.
        
        Parameters:
            training_df: Spark DataFrame after joining labels and features.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        labels_table_cfg = self.cfg.labels_table_cfg

        _logger.info('Converting training DataFrame to pandas DataFrame')
        training_set_pdf = training_df.toPandas()

        X = training_set_pdf.drop(labels_table_cfg.label_col, axis=1)
        y = training_set_pdf[labels_table_cfg.label_col]

        _logger.info(f'Splitting into train/test splits - test_size: {self.cfg.pipeline_params["test_size"]}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            random_state=self.cfg.pipeline_params['random_state'],
            test_size=self.cfg.pipeline_params['test_size'],
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def fit_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.pipeline.Pipeline:
        """
        Creates and fits an sklearn pipeline on the training data.
        
        Parameters:
            X_train: Training features as a pandas DataFrame.
            y_train: Training labels as a pandas Series.
            
        Returns:
            A fitted scikit-learn pipeline.
        """
        _logger.info('Creating sklearn pipeline...')
        pipeline = ModelTrainPipeline.create_train_pipeline(self.cfg.model_params)

        _logger.info('Fitting sklearn RandomForestClassifier...')
        _logger.info(f'Model params: {pprint.pformat(self.cfg.model_params)}')
        model = pipeline.fit(X_train, y_train)

        return model

    def run(self):
        """
        Executes the model training process and logs all parameters, metrics, and artifacts to MLflow.
        
        Steps:
          1. Set the MLflow experiment.
          2. Start an MLflow run.
          3. Read and join data from the feature delta table and labels table.
          4. Create train/test splits.
          5. Fit the sklearn pipeline.
          6. Log the model using mlflow.sklearn.log_model.
          7. Compute evaluation metrics manually and log them.
          8. Optionally register the model in MLflow Model Registry.
        """
        _logger.info('========== Running model training ==========')
        mlflow_tracking_cfg = self.cfg.mlflow_tracking_cfg

        _logger.info('========== Setting MLflow experiment ==========')
        self._set_experiment(mlflow_tracking_cfg)
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        _logger.info('========== Starting MLflow run ==========')
        with mlflow.start_run(run_name=mlflow_tracking_cfg.run_name) as mlflow_run:
            if self.cfg.conf is not None:
                mlflow.log_dict(self.cfg.conf, 'conf.yml')
            if self.cfg.env_vars is not None:
                mlflow.log_dict(self.cfg.env_vars, 'env_vars.yml')

            _logger.info('========== Reading and preparing training data ==========')
            training_df = self.get_training_data()

            _logger.info('========== Creating train/test splits ==========')
            X_train, X_test, y_train, y_test = self.create_train_test_split(training_df)

            _logger.info('========== Fitting RandomForestClassifier model ==========')
            model = self.fit_pipeline(X_train, y_train)

            _logger.info('Logging model to MLflow using mlflow.sklearn.log_model')
            mlflow.sklearn.log_model(
                model,
                artifact_path='model',
                input_example=X_train[:100],
                signature=infer_signature(X_train, y_train)
            )

            _logger.info('========== Model Evaluation ==========')
            _logger.info('Evaluating and logging metrics individually')
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='binary')
            recall = recall_score(y_test, predictions, average='binary')
            f1 = f1_score(y_test, predictions, average='binary')

            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)

            metrics = {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1
            }
            print(pd.DataFrame(metrics, index=[0]))

            if mlflow_tracking_cfg.model_name is not None:
                _logger.info('========== MLflow Model Registry ==========')
                _logger.info(f'Registering model: {mlflow_tracking_cfg.model_name}')
                mlflow.register_model(f'runs:/{mlflow_run.info.run_id}/model', name=mlflow_tracking_cfg.model_name)

        _logger.info('========== Model training completed ==========')
