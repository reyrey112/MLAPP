import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Docstring for IngestData
    Ingesting data from the data path

    """
    def __init__(self, data_path: str) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        :param data_path: path to the datapath
        :type data_path: str
        """
        self.data_path = data_path

    def get_data(self):
        """
        Docstring for get_data
        Ingesting the data from the data_path

        :param self: Description
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

@step
def ingest_df(data_path:str) -> pd.DataFrame:
    """
    Docstring for ingest_data
    Ingesting the data from the data_path
    
    :param data_path: path to the data
    :type data_path: str
    :return: the ingested data
    :rtype: DataFrame
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"error while ingesting data: {e}")
        raise e