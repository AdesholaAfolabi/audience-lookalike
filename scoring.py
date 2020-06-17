from load_attributes import Attributes
from feature_engineering import FeatureEngineering
import pickle
import scipy as sci
import pandas as pd


class score_users(Attributes):
    """ 
    This is where the scoring of users happen based on their
    distance to clusters (our varticals and campaign types)
       
    """

    def __init__(self, path, preprocessor= None, model_pickle= None):
        
        Attributes.__init__(self, path)

        self.preprocessor = preprocessor #This saves the pipeline object not initially declared in the Attributes init function.
        self.model_pickle = model_pickle #This saves the model object not initially declared in the Attributes init function.
        self.result = dict()
        
    def print_data_shape(self):
        
        """
        
        Function to check if inheritance works properly. Confirms that self.data
        is inherited from the attribute class
                
        """
        
        self.read_yaml_file()
        self.read_csv()
        print (self.data.head())
        print ("The shape of the dataset is {}".format(self.data.shape))
    
    
    def load_processor(self, processor_path):
        
        """
        
        Function to load the pipeline pickle file
                
        Args:
            None
        
        Returns:
            The pipeline object
        
        """
        
        with open(processor_path, 'rb') as file:
            self.preprocessor = pickle.load(file)
        return self.preprocessor
    
    def load_model(self, model_path):
        
        """
        
        Function to load the model pickle file
                
        Args:
            None
        
        Returns:
            The model object
        
        """
        
        with open(model_path, 'rb') as file:
            self.model_pickle = pickle.load(file)
        return self.model_pickle
    
    def fill_na(self): #This function handles NaN values in the dataset
        
        """
        
        Function handles NaN values in a dataset for both categorical
        and numerical variables
    
        
        """
        
        for item in self.data[self.num]:
            self.data[item] = self.data[item].fillna(self.data[item].mean())
        for item in self.data[self.cat]:
            self.data[item] = self.data[item].fillna(self.data[item].value_counts().index[0])
            #self.data[item] = self.data[item].fillna(method = 'ffill')
        

    def score(self):
        
        """
        
        Function to score users based on their distance to clusters. Starts by reading the
        YAML file for necessary input columns and goes on to read the file. The model
        and preprocessor artifact are also loaded and data is eventually returned back to a 
        dataframe which is saved in an output path.
                
        Args:
            None
        
        Returns:
            Final DataFrame that contains MSISDNs and their distance to clusters
        
        """
        
        self.read_yaml_file()
        self.read_csv()
        self.load_processor('pipeline.pkl')
        self.load_model('kmeans.pkl')
        self.msisdn = self.data['msisdn']
        self.fill_na()
        self.data.drop(['msisdn'],axis=1,inplace=True)
        self.data_score = self.preprocessor.transform(self.data)
        self.data_score = self.data_score.toarray()
        self.predictions = self.model_pickle.predict(self.data_score)
        
        self.center = self.model_pickle.cluster_centers_[0]
        
        #self.result = dict()
        for i, j in zip(self.data_score, self.msisdn):
            self.result[j] = sci.spatial.distance.euclidean(self.center , i)
            
        self.final_result = pd.DataFrame(self.result.items(), columns=['msisdn', 'score'])
        return self.final_result
        