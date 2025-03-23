import unittest
from pyspark.sql import SparkSession
import yaml
from booking.common import MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from booking.model_train import ModelTrain, ModelTrainConfig

class TestCreateTrainTestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("TestCreateTrainTestSplit") \
            .getOrCreate()

    def setUp(self):
        """
        Prepare a minimal ModelTrain instance with minimal config 
        so we can specifically test create_train_test_split.
        """
        mlflow_cfg = MLflowTrackingConfig(
            experiment_id=None,            
            experiment_path="/dummy_path",  
            run_name="dummy_run",
            model_name=None                
        )
        feature_store_cfg = FeatureStoreTableConfig(
            database_name="dummy_db",
            table_name="dummy_table",
            primary_keys=["id"],
            description="dummy description"
        )
        labels_cfg = LabelsTableConfig(
            database_name="dummy_db",
            table_name="dummy_label_table",
            label_col="label",
            dbfs_path="dbfs:/dummy_path"
        )

        self.model_train_cfg = ModelTrainConfig(
            mlflow_tracking_cfg=mlflow_cfg,
            feature_store_table_cfg=feature_store_cfg,
            labels_table_cfg=labels_cfg,
            pipeline_params={
                "test_size": 0.5,       
                "random_state": 42      
            },
            model_params={}        
        )

        self.model_train = ModelTrain(cfg=self.model_train_cfg)

    def test_create_train_test_split(self):
        """
        Test that create_train_test_split properly converts Spark DataFrame to pandas,
        and splits into train/test sets according to test_size, random_state, etc.
        """
        data = [
            (1, 0, 10.0),
            (2, 1, 20.0),
            (3, 0, 30.0),
            (4, 1, 40.0),
        ]
        schema = ["id", "label", "some_feature"]
        spark_df = self.spark.createDataFrame(data, schema=schema)
        X_train, X_test, y_train, y_test = self.model_train.create_train_test_split(spark_df)

        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

        self.assertNotIn("label", X_train.columns)
        self.assertNotIn("label", X_test.columns)

if __name__ == "__main__":
    unittest.main()
