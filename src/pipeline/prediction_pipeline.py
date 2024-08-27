import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.error("Exception occurred during prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Year_Birth: float,
                 Income: float,
                 Kidhome: int,
                 Teenhome: int,
                 Recency: int,
                 MntWines: float,
                 MntFruits: float,
                 MntMeatProducts: float,
                 MntFishProducts: float,
                 MntSweetProducts: float,
                 MntGoldProds: float,
                 NumDealsPurchases: int,
                 NumWebPurchases: int,
                 NumCatalogPurchases: int,
                 NumStorePurchases: int,
                 NumWebVisitsMonth: int,
                 AcceptedCmp3: int,
                 AcceptedCmp4: int,
                 AcceptedCmp5: int,
                 AcceptedCmp1: int,
                 AcceptedCmp2: int,
                 Complain: int,
                 Z_CostContact: int,
                 Z_Revenue: int,
                 Education: str,
                 Marital_Status: str):
        
        self.Year_Birth = Year_Birth
        self.Income = Income
        self.Kidhome = Kidhome
        self.Teenhome = Teenhome
        self.Recency = Recency
        self.MntWines = MntWines
        self.MntFruits = MntFruits
        self.MntMeatProducts = MntMeatProducts
        self.MntFishProducts = MntFishProducts
        self.MntSweetProducts = MntSweetProducts
        self.MntGoldProds = MntGoldProds
        self.NumDealsPurchases = NumDealsPurchases
        self.NumWebPurchases = NumWebPurchases
        self.NumCatalogPurchases = NumCatalogPurchases
        self.NumStorePurchases = NumStorePurchases
        self.NumWebVisitsMonth = NumWebVisitsMonth
        self.AcceptedCmp3 = AcceptedCmp3
        self.AcceptedCmp4 = AcceptedCmp4
        self.AcceptedCmp5 = AcceptedCmp5
        self.AcceptedCmp1 = AcceptedCmp1
        self.AcceptedCmp2 = AcceptedCmp2
        self.Complain = Complain
        self.Z_CostContact = Z_CostContact
        self.Z_Revenue = Z_Revenue
        self.Education = Education
        self.Marital_Status = Marital_Status

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Year_Birth': [self.Year_Birth],
                'Income': [self.Income],
                'Kidhome': [self.Kidhome],
                'Teenhome': [self.Teenhome],
                'Recency': [self.Recency],
                'MntWines': [self.MntWines],
                'MntFruits': [self.MntFruits],
                'MntMeatProducts': [self.MntMeatProducts],
                'MntFishProducts': [self.MntFishProducts],
                'MntSweetProducts': [self.MntSweetProducts],
                'MntGoldProds': [self.MntGoldProds],
                'NumDealsPurchases': [self.NumDealsPurchases],
                'NumWebPurchases': [self.NumWebPurchases],
                'NumCatalogPurchases': [self.NumCatalogPurchases],
                'NumStorePurchases': [self.NumStorePurchases],
                'NumWebVisitsMonth': [self.NumWebVisitsMonth],
                'AcceptedCmp3': [self.AcceptedCmp3],
                'AcceptedCmp4': [self.AcceptedCmp4],
                'AcceptedCmp5': [self.AcceptedCmp5],
                'AcceptedCmp1': [self.AcceptedCmp1],
                'AcceptedCmp2': [self.AcceptedCmp2],
                'Complain': [self.Complain],
                'Z_CostContact': [self.Z_CostContact],
                'Z_Revenue': [self.Z_Revenue],
                'Education': [self.Education],
                'Marital_Status': [self.Marital_Status]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered successfully')
            return df
        except Exception as e:
            logging.error('Exception occurred while creating DataFrame')
            raise CustomException(e, sys)
