import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import read_sql_data
#from src.components.data_ingestion import DataIngestion
from sklearn.model_selection import train_test_split

from dataclasses import dataclass




@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    row_data_path:str = os.path.join('artifacts','row.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            ## reading data from mysql
            df = read_sql_data()

            logging.info("Reading  completed data from My_sql")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.row_data_path,index=False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("DATA INGESTION IS COMPLETED oK!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            