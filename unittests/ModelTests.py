#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model = model_load()
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load()
    
        ## ensure that a list can be passed
        query = {'country': ['united_states','singapore','united_states'],
                 'age': [24,42,20],
                 'subscriber_type': ['aavail_basic','aavail_premium','aavail_basic'],
                 'num_streams': [8,17,14]
        }

        result = model_predict(query,model,test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] in [0,1])

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
