import pyspark.sql.dataframe
from pyspark.sql import SparkSession
import mlflow
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
    def __init__(self, model_uri: str, input_table_name: str, feature_table_name: str, lookup_keys: list, output_table_name: str = None):
        """
        Parameters
        ----------
        model_uri : str
            MLflow model URI. The model must have been logged using mlflow.sklearn.log_model.
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
        and applies it to perform predictions.
        
        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Input DataFrame containing lookup keys.
        
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            A Spark DataFrame with all original columns and an added 'prediction' column.
        """
        _logger.info(f"Loading feature table: {self.feature_table_name}")
        feature_df = spark.table(self.feature_table_name)

        _logger.info(f"Joining input DataFrame with feature table on keys: {self.lookup_keys}")
        joined_df = df.join(feature_df, on=self.lookup_keys, how='inner')

        _logger.info("Converting joined DataFrame to pandas for scoring")
        pandas_df = joined_df.toPandas()

        _logger.info(f"Loading model from MLflow: {self.model_uri}")
        model = mlflow.sklearn.load_model(self.model_uri)

        _logger.info("Performing predictions using the loaded model")
        predictions = model.predict(pandas_df)

        _logger.info("Appending predictions to the DataFrame")
        pandas_df['prediction'] = predictions

        _logger.info("Converting pandas DataFrame back to Spark DataFrame")
        return spark.createDataFrame(pandas_df)

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
