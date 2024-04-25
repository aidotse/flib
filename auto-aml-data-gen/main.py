from simulate import init_params, create_param_files, run_simulation
from preprocess import preprocess
from classifier import Classifier
from optimizer import Optimizer

def main(n_trials:int=10, operating_recall:float=0.8, target:float=0.95):
    
    #params = init_params(seed=0)
    #param_file = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
    #create_param_files(params, param_file)
    #run_simulation(param_file)
    #path_to_tx_log = '/home/edvin/Desktop/flib/AMLsim/outputs/tmp/tx_log.csv'
    #datasets = preprocess(path_to_tx_log, ['bank'], 0.0)
    #trainset, testset = datasets[0]
    #print(f'trainset - n samples: {trainset.shape[0]}, label ratio: {trainset[trainset["is_sar"]==1].shape[0]/trainset[trainset["is_sar"]==0].shape[0]:.4f}')
    #print(f'testset - n samples: {testset.shape[0]}, label ratio: {testset[testset["is_sar"]==1].shape[0]/testset[testset["is_sar"]==0].shape[0]:.4f}')
    #classifier = Classifier(dataset=(trainset, testset))
    #model = classifier.train(model='RandomForestClassifier', tune_hyperparameters=True)
    #fpr, importances = classifier.evaluate(operating_recall=operating_recall)
    
    optimizer = Optimizer(target=0.95, max=0.92, operating_recall=operating_recall)
    best_trials = optimizer.optimize(n_trials=n_trials)
    print(f'best value: {best_trials}')
    
    return









if __name__ == '__main__':
    iterations = 3
    ratio = 0.05
    operating_recall = 0.8
    target = 0.95
    main(iterations, operating_recall, target)
