from simulate import init_params, create_param_files, run_simulation
from preprocess import preprocess
from train import train, evaluate, Classifier

def main(iterations:int=10):
    
    params = init_params(seed=0)
    param = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
    create_param_files(params, param)
    run_simulation(param)
    path_to_tx_log = '/home/edvin/Desktop/flib/AMLsim/outputs/tmp/tx_log.csv'
    datasets = preprocess(path_to_tx_log, ['handelsbanken'], 0.9)
    dataset = datasets[0]
    
    print(dataset[0].head())
    
    #model = train(datasets[0][0])
    #evaluate(model, a)
    
    classifier = Classifier(dataset)
    
    return









if __name__ == '__main__':
    ratio = 0.05
    recall = 0.8
    fpr = 0.95
    main()
