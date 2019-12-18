import time,os,re,csv,sys,uuid,joblib
import pickle
from datetime import date
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from logger import update_predict_log, update_train_log

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".","models")):
    os.mkdir("models") 

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "SVM on AAVAIL churn"
SAVED_MODEL = os.path.join("models","model-{}.joblib".format(re.sub("\.","_",str(MODEL_VERSION))))


def load_aavail_data():
    data_dir = os.path.join(".","data")
    df = pd.read_csv(os.path.join(data_dir,r"aavail-target.csv"))
       
    ## pull out the target and remove uneeded columns
    _y = df.pop('is_subscriber')
    y = np.zeros(_y.size)
    y[_y==0] = 1 
    df.drop(columns=['customer_id','customer_name'],inplace=True)
    df.head()
    X = df

    return(X,y)

def get_preprocessor():
    """
    return the preprocessing pipeline
    """

    ## preprocessing pipeline
    numeric_features = ['age', 'num_streams']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])

    categorical_features = ['country', 'subscriber_type']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])


    return(preprocessor)

def model_train(test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    The iris dataset is already small so the subset is shown as an example

    Note that the latest training data is always saved to be used by perfromance monitoring tools.
    """

    ## start timer for runtime
    time_start = time.time()
    
    ## data ingestion
    X,y = load_aavail_data()


    preprocessor = get_preprocessor()

    ## subset the data to enable faster unittests
    if test:
        n_samples = int(np.round(0.9 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]  
    
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    ## Specify parameters and model
    param_grid = {
        'clf__n_estimators': [25,50,75,100],
        'clf__criterion':['gini','entropy'],
        'clf__max_depth':[2,4,6]
    }

    print("... grid searching")
    clf = ensemble.RandomForestClassifier()
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    params = grid.best_params_
    params = {re.sub("clf__","",key):value for key,value in params.items()}
    
    ## fit model on training data
    clf = ensemble.RandomForestClassifier(**params)
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    eval_test = classification_report(y_test,y_pred,output_dict=True)
    
    ## retrain using all data
    pipe.fit(X, y)

    if test:
        print("... saving test version of model")
        joblib.dump(pipe,os.path.join("models","test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        joblib.dump(pipe,SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models",'latest-train.pickle')
        with open(data_file,'wb') as tmp:
            pickle.dump({'y':y,'X':X},tmp)
        
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_train_log(X.shape,eval_test,runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=test)

def model_predict(query,model=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()
    
    ## input checks
    if isinstance(query,dict):
        query = pd.DataFrame(query)
    elif isinstance(query,pd.DataFrame):
        pass
    else:
        raise Exception("ERROR (model_predict) - invalid input. {} was given".format(type(query)))

    ## features check
    features = sorted(query.columns.tolist())
    if features != ['age', 'country', 'num_streams', 'subscriber_type']:
        print("query features: {}".format(",".join(features)))
        raise Exception("ERROR (model_predict) - invalid features present") 
    
    ## load model if needed
    if not model:
        model = model_load()
    
    ## output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = 'None'
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    for i in range(query.shape[0]):
        update_predict_log(y_pred[i],y_proba,query.iloc[i].values.tolist(), 
                           runtime,MODEL_VERSION,test=test)
        
    return({'y_pred':y_pred,'y_proba':y_proba})


def model_load():
    """
    example funtion to load model
    """

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)


if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    
    ## train the model
    model_train(test=True)

    ## load the model
    model = model_load()
    
    ## example predict
    query = pd.DataFrame({'country': ['united_states','singapore','united_states'],
                          'age': [24,42,20],
                          'subscriber_type': ['aavail_basic','aavail_premium','aavail_basic'],
                          'num_streams': [8,17,14]
    })

    result = model_predict(query,model,test=True)
    y_pred = result['y_pred']
    print("predicted: {}".format(y_pred))
