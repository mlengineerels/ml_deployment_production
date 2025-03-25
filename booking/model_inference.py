import pyspark.sql.dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import mlflow
import mlflow.pyfunc
from booking.utils.logger_utils import get_logger

_logger = get_logger()

# Create a SparkSession for inference
spark = SparkSession.builder.appName('modelinference').getOrCreate()

class ModelInference:
    """
    Class to execute model inference.
    Applies the model at the specified URI for batch inference on the input table,
    joining with feature data from a Delta table based on lookup keys.
    The results are written to the output table if specified.
    """
    def __init__(
        self, 
        model_uri: str, 
        input_table_name: str, 
        feature_table_name: str, 
        lookup_keys: list, 
        output_table_name: str = None
    ):
        """
        Parameters
        ----------
        model_uri : str
            MLflow model URI. The model must have been logged using mlflow.sklearn.log_model or as a pyfunc.
        input_table_name : str
            Table name to load as a Spark DataFrame containing lookup keys for joining feature data.
        feature_table_name : str
            Delta table name containing the feature data.
        lookup_keys : list
            List of column names to use as join keys between the input table and the feature table.
        output_table_name : str, optional
            Output table name to write predictions to.
        """
        self.model_uri = model_uri
        self.input_table_name = input_table_name
        self.feature_table_name = feature_table_name
        self.lookup_keys = lookup_keys
        self.output_table_name = output_table_name

    def _load_input_table(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Loads the input table as a Spark DataFrame.
        
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            Input DataFrame containing lookup keys.
        """
        _logger.info(f"Loading input table: {self.input_table_name}")
        return spark.table(self.input_table_name)

    def score_batch(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        """
        Joins the input DataFrame with the feature table on lookup keys, loads the model from MLflow,
        and applies it to perform predictions using an MLflow pyfunc Spark UDF.
        
        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Input DataFrame containing lookup keys.
        
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            A Spark DataFrame with all original columns plus an added 'prediction' column.
        """
        _logger.info(f"Loading feature table: {self.feature_table_name}")
        feature_df = spark.table(self.feature_table_name)

        _logger.info(f"Joining input DataFrame with feature table on keys: {self.lookup_keys}")
        joined_df = df.join(feature_df, on=self.lookup_keys, how='inner')

        _logger.info(f"Loading model from MLflow as a Spark UDF: {self.model_uri}")
        # Load model as PyFunc UDF; make sure to specify output data type (DoubleType here as an example).
        pyfunc_udf = mlflow.pyfunc.spark_udf(
            spark,
            model_uri=self.model_uri,
            result_type=DoubleType()
        )

        # ------------------------------------------------------------
        # Identify the exact columns the model expects, in the order
        # it expects them. For illustration, we assume all columns
        # except the lookup keys are needed for prediction.
        # Adjust as necessary to match your training columns/ordering.
        # ------------------------------------------------------------
        feature_columns = [c for c in joined_df.columns if c not in self.lookup_keys]

        _logger.info("Applying the MLflow pyfunc UDF to perform predictions in Spark")
        # Pass the feature columns to the UDF in the exact order the model was trained on
        predicted_df = joined_df.withColumn(
            "prediction",
            pyfunc_udf(*[col(c) for c in feature_columns])
        )

        return predicted_df

    def run_batch(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Runs batch inference by loading input data, joining with feature data, and scoring the model.
        
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            A DataFrame containing the original input data, feature values, and a 'prediction' column.
        """
        input_df = self._load_input_table()
        pred_df = self.score_batch(input_df)
        return pred_df

    def run_and_write_batch(self, mode: str = 'overwrite') -> None:
        """
        Executes batch inference and writes the predictions to a Delta table.
        
        Parameters
        ----------
        mode : str, optional
            Write mode. Options include:
                - "append": Append to existing data.
                - "overwrite": Overwrite existing data.
        """
        _logger.info("========== Running batch model inference ==========")
        pred_df = self.run_batch()

        if self.output_table_name is None:
            raise ValueError("output_table_name must be specified to write predictions.")

        _logger.info("========== Writing predictions ==========")
        _logger.info(f"Write mode: {mode}")
        _logger.info(f"Writing predictions to table: {self.output_table_name}")

        pred_df.write.format("delta").mode(mode).saveAsTable(self.output_table_name)

        _logger.info("========== Batch model inference completed ==========")
