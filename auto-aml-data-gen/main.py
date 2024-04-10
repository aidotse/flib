from simulate import init_params, create_param_files, run_simulation
from preprocess import preprocess
from classifier import Classifier

def main(iterations:int=10, operating_recall:float=0.8):
    
    params = init_params(seed=0)
    param = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
    create_param_files(params, param)
    run_simulation(param)
    path_to_tx_log = '/home/edvin/Desktop/flib/AMLsim/outputs/tmp/tx_log.csv'
    datasets = preprocess(path_to_tx_log, ['bank'], 0.0)
    dataset = datasets[0]
    
    print(dataset[0].head())
    classifier = Classifier(dataset=dataset)
    model = classifier.train(model='RandomForestClassifier', tune_hyperparameters=True)
    fpr = classifier.evaluate(operating_recall=operating_recall)
    
    print('hej')
    
    return









if __name__ == '__main__':
    iterations = 10
    ratio = 0.05
    operating_recall = 0.8
    fpr = 0.95
    main(iterations, operating_recall)
