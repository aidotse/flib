import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from typing import List, Tuple

# stop warnings
import warnings
warnings.filterwarnings("ignore")

def calculate_AP(model, X_test, y_test):
    y_score = model.predict_proba(X_test)[:,1]
    average_precision = average_precision_score(y_test, y_score)
    return average_precision

def calculate_ROC_AUC(model, X_test, y_test):
    y_score = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_score)
    return roc_auc

def calculate_tnfpfntp(model, X_test, ground_truths, thresholds=[0.5]):
    predictions = model.predict_proba(X_test)[:,1]
    tnfpfntp = {}
    for i, threshold in enumerate(thresholds):
        prediction = (predictions >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(ground_truths, prediction).ravel()
        tnfpfntp[threshold] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    return tnfpfntp

def get_dataset(dataset):

    trainset, testset = dataset
    if type(trainset) == str and type(testset) == str:
        trainset = pd.read_csv(trainset)
        testset = pd.read_csv(testset)
    elif type(trainset) == pd.DataFrame and type(testset) == pd.DataFrame:
        pass
    else:
        raise ValueError('trainset and testset must be either paths to csv files or pandas DataFrames')

    # If column 'true_label' exists, remove it
    if 'true_label' in trainset.columns:
        X_train = trainset.drop(['is_sar','account','bank','true_label'], axis=1)
    else:
        X_train = trainset.drop(['is_sar','account','bank'], axis=1)
    y_train = trainset[['is_sar']]
    X_test = testset.drop(['is_sar','account','bank'], axis=1)
    y_test = testset[['is_sar']]

    # Removing rows with missing labels
    X_train = X_train[y_train['is_sar'] != -1]
    y_train = y_train[y_train['is_sar'] != -1]

    # In X_train, replace all NaN values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)


    return X_train, y_train, X_test, y_test


def train_model(dataset, model):
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    model.fit(X_train, y_train)
    tnfpfntp = calculate_tnfpfntp(model, X_test, y_test, np.round(np.linspace(0,1,101), 2))
    return tnfpfntp

def train_models(dataset, model_names:list=['XGB','RF','SVM','KNN','LOG']):
    
    results = {}
    
    for model_name in tqdm(model_names):
        if model_name == 'SVM':
            model = svm.SVC(kernel='rbf',probability=True)
        elif model_name == 'KNN':
            model = KNeighborsClassifier(metric='euclidean', n_neighbors=43, weights='distance')
        elif model_name == 'LOG':
            model = LogisticRegression(C=10, max_iter=1000, penalty='l1', solver='liblinear') 
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=5) 
        elif model_name == 'XGB':
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5,min_samples_leaf=5, min_samples_split=5)
        
        tnfpfntp = train_model(dataset, model)
        results[model_name] = tnfpfntp
    
    return results


def main():

    GROUND_DATASET = 'data_NEW' # Name of ground dataset
    RESULTS_FOLDER = 'Results' # Name of results folder

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    model_names = ['XGB','RF','SVM','KNN','LOG'] # Put what models as a list
    datasets = ['100K_accts_EASY25_NEW_NEW','100K_accts_MID5_NEW_NEW','100K_accts_HARD1_NEW_NEW'] # Put datasets as a list
    ratios = ['no_noise','0.1','0.25'] # What ratios to train (subfolders in datasets)

    results_df = pd.DataFrame(columns=['model', 'dataset', 'ratio', 'noise', 'AP','ROC_AUC'])

    for model_name in model_names:
        for dataset in datasets:
            for ratio in ratios:
                for noise in os.listdir(f'{GROUND_DATASET}/{dataset}/bank/train/{ratio}'):
                    if model_name == 'SVM':
                        model = svm.SVC(kernel='rbf',probability=True)
                    elif model_name == 'KNN':
                        model = KNeighborsClassifier(metric= 'euclidean', n_neighbors= 43, weights= 'distance')
                    elif model_name == 'LOG':
                        model = LogisticRegression(C= 10, max_iter= 1000, penalty= 'l1', solver= 'liblinear') 
                    elif model_name == 'RF':
                        model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=5) 
                    elif model_name == 'XGB':
                        if dataset == '100K_accts_EASY25_NEW_NEW':
                            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,min_samples_leaf=2, min_samples_split=2)
                        else:
                            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5,min_samples_leaf=5, min_samples_split=5)
                    
                    trainset_path = f'{GROUND_DATASET}/{dataset}/bank/train/{ratio}/{noise}'
                    testset_path = f'{GROUND_DATASET}/{dataset}/bank/test/nodes.csv'
                    datasets_path = (trainset_path, testset_path)
                    print(f'Training model {model_name} on {dataset} with ratio {ratio} and noise {noise}')
                    AP,ROC_AUC = train_model(model, datasets_path)
                    
                    new_results = pd.DataFrame({'model': model_name, 'dataset': dataset, 'ratio': ratio, 'noise': noise, 'AP': AP,'ROC_AUC': ROC_AUC}, index=[0])
                    results_df = pd.concat([results_df, new_results], ignore_index=True)
                    results_df.to_csv(f'{RESULTS_FOLDER}/{model_name}.csv', index=False)

if __name__ == '__main__':
    main()    
