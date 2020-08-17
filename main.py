from load_attributes import Attributes
from scoring import score_users
import os
import s3fs, pandas as pd,os
fs = s3fs.S3FileSystem()

class demography(Attributes):
    
    """ 
    
    This class here has the main function where scoring happens. 
    It uses the score_users module to score all the users and also
    inherits all the attributes from the load_attributes module.
    
    """
    
    def __init__(self, path):
        
        Attributes.__init__(self, path)
        self.files = []
        
    def get_data_list(self):
        
        for file in fs.ls(self.path):
            self.files.append(file)
        
    def score_demography(self):
        
        """
        
        Function to score the entire flat demography. Calls the scoring module
                
        Args:
            None
        
        Returns:
            The dataframe of the lookalike scores to the predictions folder.
        
        """
        
        counter = 0
        self.get_data_list()
        for file in self.files:
            self.test = score_users("s3://"+file)
            self.dataframe = self.test.score()
            out = 'out'+ str(counter) + '.csv'
            bytes_write = self.dataframe.to_csv(None,index=False,sep=',').encode()
            with fs.open("s3://datateam-ml/new_lookalike/installs_and_downloads/{}".format(out), 'wb') as f:
                f.write(bytes_write)
            
            counter += 1
            print ('Done with the lookalike scoring')