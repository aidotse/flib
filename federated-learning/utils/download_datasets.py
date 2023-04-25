import pandas as pd

def download_adult_dataset(save_path:str='adult.csv'):
    names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=names)
    data.to_csv(path_or_buf=save_path, index=False)

download_adult_dataset('data/adult/adult.csv')