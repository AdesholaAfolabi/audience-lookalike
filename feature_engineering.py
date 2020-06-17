from load_attribributes import Attributes
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

class FeatureEngineering(Attributes):
    
    """ This class is for all preprocessing tasks that is required for this project """
    
    def __init__(self, path):
        Attributes.__init__(self, path)
        
    def print_data_shape(self):
        
        """
        
        Function to check if inheritance works properly. Confirms that self.data
        is inherited from the attribute class
                
        """
        
        self.read_yaml_file()
        self.read_csv()
        print (self.data.head())
        print ("The shape of the dataset is {}".format(self.data.shape))
        
    def fill_na(self): #This function handles NaN values in the dataset
        
        """
        
        Function handles NaN values in a dataset for both categorical
        and numerical variables
    
        
        """
        
        for item in self.data[self.num]:
            self.data[item] = self.data[item].fillna(self.data[item].mean())
        for item in self.data[self.cat]:
            self.data[item] = self.data[item].fillna(self.data[item].value_counts().index[0])
            
    def hash_list(self): 
        
        """
        
        Function creates a sparse matrix of the categorical 
        variables contained in the dataset
    
        """
        
        self.hash_features = []
        for item in self.cat:
            if item not in self.low_cat:
                self.hash_features.append(item)
                
    def pipeline(self, hash_size):
        
        """  
         Function contains the object to handle all scenarios in the dataset.
         It is broken down into numerical, categorical and hash pipeline
                
        Args:
            hash_size: specifies the hash bucket size to be used
        """
        
        self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
        self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                       ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
        self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
        
    
    def build_pipe(self, hash_size = 100):
        
        """
        
        Function that builds the pipeline and returns the 
        pipeline object and the data to be used for modeling
                
        Args:
            hash_bucket size
        
        Returns:
            pipeline object
            data to be used for training after being transformed by the pipeline
        
        """
        
        self.read_yaml_file()
        self.read_csv()
        self.fill_na()
        self.data.drop(['msisdn'],axis=1,inplace=True)
        self.hash_list()
        self.pipeline(hash_size)
        
        self.full_pipeline = ColumnTransformer(
        transformers=[
            ('num', self.num_pipeline, self.num),
            ('cat', self.cat_pipeline, self.low_cat),
            ('hash', self.hash_pipeline, self.hash_features)
        ])
        
        self.X = self.data
        
        self.full_pipeline.fit(self.X)
        
        self.X = self.full_pipeline.transform(self.X)
        
        print(self.X.shape)
        return self.X, self.full_pipeline