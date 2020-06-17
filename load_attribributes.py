import yaml
import pandas as pd

class Attributes:
    
    def __init__(self, path):
    
        """ 
        This class takes in the yaml file, reads the different
        attributes and initializes the initial set of variables
        
        """
        self.path = path
        self.input_col = []
        self.num = []
        self.cat = []
        self.low_cat = []
        self.data = pd.DataFrame()

    def read_yaml_file(self):
    
        """Function to read in the yaml file. The file should have
        a list of attributes to be used. The columns are stored 
        in the attributes list.
                
        Args:
            None
        
        Returns:
            None
        
        """
        
        features = yaml.safe_load(open("features.yml"))
        self.input_col = features['input_col']
        self.num = features['num_features']
        self.cat = features['cat_features']
        self.low_cat = features['low_cat']
        print ("Input column is {}".format(self.input_col))
        
    def read_csv(self):
        
        """
        
        This funtion takes in a file path - csv & parquet and reads the data
        based on the input columns specified 
        
        Returns:
            dataset to be used for training
        
        """
        if self.path.endswith('.csv'):
            data = pd.read_csv(self.path, usecols = self.input_col)
            print('CSV file read sucessfully')
            data = data.reindex(columns = self.input_col)
            self.data = data
            return data
            
        elif self.path.endswith('.parquet'):
            data = pd.read_parquet(self.path, engine = 'pyarrow', columns = self.input_col)
            print('parquet file read sucessfully')
            data.columns = data.columns.astype(str)
            data = data.reindex(columns = self.input_col)
            self.data = data
            return self.data
        
        else:
            return ('No CSV file or Parquet file was passed')