from feature_engineering import FeatureEngineering
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy as sci

class lookalike:
    """ 
    
    This class is a description of where the model is built. 
    The intention here is to build a lookalike model using K-Means
    
    """

    def __init__(self, data_path, data=None, model=None, pipeline=None):
        
        self.model = model
        self.pipeline = pipeline
        self.data_path = data_path
        self.data = data
        
        
    def get_data_and_pipeline(self):
        
        """
        
        Function to obtain the pipeline object from the feature engineering class
        imported above among the libraries
                
        Args:
            None
        
        Returns:
            None
        
        """
        
        data = FeatureEngineering(self.data_path)
        X, full_pipeline = data.build_pipe(hash_size=100)
        self.data = X
        self.pipeline = full_pipeline
        
    def build_model(self):
        
        """
        
        Function to build the KMeans clustering algorithm starting out with two clusters
        The model updates the init model created above
                
        Args:
            None
        
        Returns:
            None
        
        """
        
        model = KMeans(n_clusters=2, random_state=0)
        kmeans = model.fit(self.data)
        self.model = kmeans

    def save_model_and_pipeline(self, model_name=None, pipeline_name =None): 
        
        """
        
        Function saves the KMeans and Pipeline objects
                
        Args:
            model_name to be used to save the pickle file
            pipeline name to be used to save the pickle file
        
        Returns:
            None
        
        """
        
        if pipeline_name:
            pickle.dump(self.pipeline, open(pipeline_name, 'wb'))
        elif model_name:
            pickle.dump(self.model, open(model_name, 'wb'))
        else:
            pass
        
    def complile_functions(self):
        
        """
        This funtion compiles all the other funtions created above.
        """
        
        self.get_data_and_pipeline()
        self.build_model()
        self.save_model_and_pipeline(pipeline_name = "pipeline.pkl", model_name = "kmeans.pkl")