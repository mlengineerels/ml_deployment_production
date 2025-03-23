from typing import Union, List
import pyspark
from pyspark.sql import SparkSession
def create_and_write_delta_table(df: pyspark.sql.DataFrame,
                                 table_name: str,
                                 primary_keys: Union[str, List[str]],
                                 description: str,
                                 ) -> str:
    spark = SparkSession.builder.getOrCreate()
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(table_name)

    (spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA"))

    
    (spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('comment' = '{description}')")
    )
    return table_name