from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pprint
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import sklearn
import mlflow
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient
from booking.common import MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from booking.model_train_pipeline import ModelTrainPipeline
from booking.utils.logger_utils import get_logger
import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession, DataFrame
from dataclasses import dataclass

class TestModelTrain(unittest.TestCase):
    """
    Unit tests for ModelTrain class. Demonstrates mocking Spark, MLflow calls,
    and verifying logic that compares new model metrics with production model metrics.
    """

    @classmethod
    def setUpClass(cls):
        # Create or reuse a Spark session for DataFrame references
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("TestModelTrain") \
            .getOrCreate()

    def setUp(self):
        """
        Runs before each test. Set up minimal config and objects for each scenario.
        """
        # Minimal MLflowTrackingConfig mock
        self.mlflow_tracking_cfg_mock = MagicMock()
        self.mlflow_tracking_cfg_mock.model_name = "test-model"
        self.mlflow_tracking_cfg_mock.run_name = "test-run"
        self.mlflow_tracking_cfg_mock.experiment_id = None
        self.mlflow_tracking_cfg_mock.experiment_path = "/test-experiment"

        # Minimal FeatureStoreTableConfig mock
        self.feature_store_table_cfg_mock = MagicMock()
        self.feature_store_table_cfg_mock.database_name = "fs_db"
        self.feature_store_table_cfg_mock.table_name = "fs_table"
        self.feature_store_table_cfg_mock.primary_keys = ["user_id"]

        # Minimal LabelsTableConfig mock
        self.labels_table_cfg_mock = MagicMock()
        self.labels_table_cfg_mock.database_name = "lbl_db"
        self.labels_table_cfg_mock.table_name = "lbl_table"
        self.labels_table_cfg_mock.label_col = "label"

        # Typical pipeline params
        self.pipeline_params = {
            'random_state': 42,
            'test_size': 0.3
        }
        # Simplified model params
        self.model_params = {
            'n_estimators': 5,
            'max_depth': 3,
            'random_state': 42
        }

        # Prepare ModelTrainConfig-like object
        self.model_train_cfg = MagicMock()
        self.model_train_cfg.mlflow_tracking_cfg = self.mlflow_tracking_cfg_mock
        self.model_train_cfg.feature_store_table_cfg = self.feature_store_table_cfg_mock
        self.model_train_cfg.labels_table_cfg = self.labels_table_cfg_mock
        self.model_train_cfg.pipeline_params = self.pipeline_params
        self.model_train_cfg.model_params = self.model_params
        self.model_train_cfg.conf = None
        self.model_train_cfg.env_vars = None

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.sklearn.autolog")
    @patch("mlflow.start_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    @patch("booking.model_train_pipeline.ModelTrainPipeline.create_train_pipeline")
    def test_no_production_model_promoted(
        self,
        mock_pipeline,
        mock_mlflow_client_cls,
        mock_register_model,
        mock_set_experiment,
        mock_start_run,
        mock_autolog,
        mock_log_model
    ):
        """
        If there's no existing Production model, the new model is promoted immediately.
        """
        # 1. Mock pipeline and fitted model
        mock_fitted_model = MagicMock()
        mock_fitted_model.predict.return_value = [0,1,0]
        mock_pipeline.return_value.fit.return_value = mock_fitted_model

        # 2. Mock MLflow client
        mlflow_client_instance = MagicMock()
        # Means there's no model in Production
        mlflow_client_instance.search_model_versions.return_value = []
        mock_mlflow_client_cls.return_value = mlflow_client_instance

        # 3. Mock register_model to say new model version is '2'
        mock_register_model.return_value.version = 2

        # 4. We'll mock out Spark so we don't need real data
        # Let's say get_training_data returns some Spark DataFrame
        spark_df_mock = MagicMock(spec=DataFrame)

        # We'll create minimal data for create_train_test_split
        X_train_mock = pd.DataFrame({"feat": [1,2]})
        X_test_mock = pd.DataFrame({"feat": [3]})
        y_train_mock = pd.Series([0,1])
        y_test_mock = pd.Series([1])

        from booking.model_train import ModelTrain
        model_train_obj = ModelTrain(self.model_train_cfg)

        # Patch the methods inside model_train_obj
        with patch.object(model_train_obj, 'get_training_data', return_value=spark_df_mock), \
             patch.object(model_train_obj, 'create_train_test_split', return_value=(X_train_mock, X_test_mock, y_train_mock, y_test_mock)):
            model_train_obj.run()

        # 5. Assertion: check that it was promoted to Production
        mlflow_client_instance.transition_model_version_stage.assert_called_once_with(
            name="test-model",
            version=2,
            stage="Production",
            archive_existing_versions=False
        )

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.sklearn.autolog")
    @patch("mlflow.start_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    @patch("booking.model_train_pipeline.ModelTrainPipeline.create_train_pipeline")
    def test_existing_production_model_not_promoted(
        self,
        mock_pipeline,
        mock_mlflow_client_cls,
        mock_register_model,
        mock_set_experiment,
        mock_start_run,
        mock_autolog,
        mock_log_model
    ):
        """
        If there's an existing Production model with higher accuracy, 
        the new one is not promoted.
        """
        # 1. Mock pipeline & fitted model with predictions
        mock_fitted_model = MagicMock()
        # Suppose the new model accuracy is about 0.5
        mock_fitted_model.predict.return_value = [1,0]
        mock_pipeline.return_value.fit.return_value = mock_fitted_model

        # 2. Mock MLflow client
        mlflow_client_instance = MagicMock()
        mock_mlflow_client_cls.return_value = mlflow_client_instance

        # Suppose there's a production model with accuracy=0.9
        prod_version_mock = MagicMock()
        prod_version_mock.current_stage = "Production"
        prod_version_mock.run_id = "prod_run_123"
        prod_version_mock.version = 1

        mlflow_client_instance.search_model_versions.return_value = [prod_version_mock]

        # The old production's run metrics
        old_run_data = MagicMock()
        old_run_data.data.metrics = {"test_accuracy": 0.9}
        mlflow_client_instance.get_run.return_value = old_run_data

        # The new model version is 3
        mock_register_model.return_value.version = 3

        from booking.model_train import ModelTrain
        model_train_obj = ModelTrain(self.model_train_cfg)

        # Mock get_training_data and create_train_test_split
        with patch.object(model_train_obj, 'get_training_data', return_value=MagicMock(spec=DataFrame)), \
             patch.object(model_train_obj, 'create_train_test_split', side_effect=self._mocked_create_train_test_split_new_model_worse):
            model_train_obj.run()

        # The new model should NOT be promoted if 0.5 < 0.9
        mlflow_client_instance.transition_model_version_stage.assert_not_called()

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.sklearn.autolog")
    @patch("mlflow.start_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    @patch("booking.model_train_pipeline.ModelTrainPipeline.create_train_pipeline")
    def test_existing_production_model_promoted(
        self,
        mock_pipeline,
        mock_mlflow_client_cls,
        mock_register_model,
        mock_set_experiment,
        mock_start_run,
        mock_autolog,
        mock_log_model
    ):
        """
        If there's an existing Production model with lower accuracy, 
        the new model is promoted.
        """
        # 1. Mock pipeline & fitted model
        mock_fitted_model = MagicMock()
        mock_fitted_model.predict.return_value = [1,1,1]
        mock_pipeline.return_value.fit.return_value = mock_fitted_model

        # 2. Mock MLflow client
        mlflow_client_instance = MagicMock()
        mock_mlflow_client_cls.return_value = mlflow_client_instance

        # Suppose there's a production model with accuracy=0.6
        prod_version_mock = MagicMock()
        prod_version_mock.current_stage = "Production"
        prod_version_mock.run_id = "old_prod_run"
        prod_version_mock.version = 4
        mlflow_client_instance.search_model_versions.return_value = [prod_version_mock]

        old_run_data = MagicMock()
        old_run_data.data.metrics = {"test_accuracy": 0.6}
        mlflow_client_instance.get_run.return_value = old_run_data

        # The newly registered model version is 5
        mock_register_model.return_value.version = 5

        from booking.model_train import ModelTrain
        model_train_obj = ModelTrain(self.model_train_cfg)

        # We'll patch get_training_data -> SparkDF, etc.
        with patch.object(model_train_obj, 'get_training_data', return_value=MagicMock(spec=DataFrame)), \
             patch.object(model_train_obj, 'create_train_test_split', side_effect=self._mocked_create_train_test_split_new_model_better):
            model_train_obj.run()

        mlflow_client_instance.transition_model_version_stage.assert_called_once_with(
            name="test-model",
            version=5,
            stage="Production",
            archive_existing_versions=True
        )

    # ----------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------

    def _mocked_create_train_test_split_new_model_worse(self, df):
        """
        Helper to simulate returning data such that the new model 
        ends up with a test_accuracy < existing Production's 0.9
        """
        # We'll produce predictions that yield about 0.5 test accuracy
        X_train = pd.DataFrame({"col": [1,2]})
        X_test = pd.DataFrame({"col": [3,4]})
        y_train = pd.Series([0,1])
        # intentionally create half correct, half incorrect
        y_test = pd.Series([1, 1])  # so if new model predicts [1,0], accuracy ~ 0.5
        return X_train, X_test, y_train, y_test

    def _mocked_create_train_test_split_new_model_better(self, df):
        """
        Helper to simulate returning data such that the new model 
        has a test_accuracy ~ 0.8 ( vs. old 0.6 ).
        """
        X_train = pd.DataFrame({"col": [10,20,30]})
        X_test = pd.DataFrame({"col": [40,50]})
        y_train = pd.Series([0, 1, 0])
        # So if new model consistently predicts 1, let's say we get 0.8 or something better than 0.6
        y_test = pd.Series([1, 1])
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    unittest.main()
