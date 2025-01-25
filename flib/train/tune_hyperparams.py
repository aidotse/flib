import optuna
import inspect
from flib.train import Clients
import multiprocessing
from flib.utils import set_random_seed 
from flib.train.metrics import calculate_average_precision


class HyperparamTuner():
    def __init__(self, study_name, obj_fn, seed=42, n_workers=1, storage=None, client_type=None, client_names=None, client_data=None, client_params=None): #(self, study_name, obj_fn, train_dfs, val_dfs, seed=42, device='cpu', n_workers=1, storage=None, client=None, params=None):
        self.study_name = study_name
        self.obj_fn = obj_fn
        self.seed = seed
        self.n_workers = n_workers
        self.storage = storage
        self.client_type = client_type
        self.client_names = client_names
        self.client_data = client_data
        self.client_params = client_params
    
    def objective(self, trial: optuna.Trial):
        params = {}
        for param in self.client_params['search_space']:
            if self.client_params['search_space'][param]['type'] == 'categorical':
                params[param] = trial.suggest_categorical(param, self.client_params['search_space'][param]['values'])
            elif self.client_params['search_space'][param]['type'] == 'integer':
                params[param] = trial.suggest_int(param, self.client_params['search_space'][param]['low'], self.client_params['search_space'][param]['high'])
            elif self.client_params['search_space'][param]['type'] == 'float':
                params[param] = trial.suggest_float(param, self.client_params['search_space'][param]['low'], self.client_params['search_space'][param]['high'], log=self.client_params['search_space'][param].get('log', False))
            else:
                params[param] = self.client_params['search_space'][param]    
        for param in self.client_params['default']:
            if param not in params:
                if isinstance(self.client_params['default'][param], dict):
                    params[param] = next(iter(self.client_params['default'][param]))
                    params[param+'_params'] = {}
                    for subparam in self.client_params['default'][param][params[param]]:
                        params[param+'_params'][subparam] = self.client_params['default'][param][params[param]][subparam]
                else:
                    params[param] = self.client_params['default'][param]
        client_names = []
        client_params = []
        for name, data in zip(self.client_names, self.client_data):
            client_names.append(name)
            client_params.append(data | params) 
        
        results = self.obj_fn(seed=self.seed, n_workers=self.n_workers, client_type=self.client_type, client_names=client_names, client_params=client_params)
        
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for client in results:
            round = max(results[client].keys())
            for threshold in range(0, 101):
                tpfptnfn[threshold]['tp'] += results[client][round]['val']['tpfptnfn'][threshold]['tp']
                tpfptnfn[threshold]['fp'] += results[client][round]['val']['tpfptnfn'][threshold]['fp']
                tpfptnfn[threshold]['tn'] += results[client][round]['val']['tpfptnfn'][threshold]['tn']
                tpfptnfn[threshold]['fn'] += results[client][round]['val']['tpfptnfn'][threshold]['fn']
        
        avg_pre = calculate_average_precision(tpfptnfn, (0.6, 1.0))
        
        return avg_pre
    
    def optimize(self, n_trials=10):
        # seet seed
        set_random_seed(self.seed)
        
        study = optuna.create_study(storage=self.storage, sampler=optuna.samplers.TPESampler(seed=self.seed, multivariate=True), study_name=self.study_name, direction='maximize', load_if_exists=True, pruner=optuna.pruners.HyperbandPruner())
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_trials
        
    #def optimize(self, n_trials=100, n_jobs=10):
    #    # Create the study with RDB storage for parallel processing
    #    study = optuna.create_study(storage=self.storage, 
    #                                sampler=optuna.samplers.TPESampler(), 
    #                                study_name=self.study_name, 
    #                                direction='maximize', 
    #                                load_if_exists=True,
    #                                pruner = optuna.pruners.HyperbandPruner())
    #    
    #    
    #    def run_study():
    #        study.optimize(self.objective, n_trials=n_trials // n_jobs)
    #
    #    processes = []
    #    for _ in range(n_jobs):
    #        p = multiprocessing.Process(target=run_study)
    #        p.start()
    #        processes.append(p)
    #
    #    for p in processes:
    #        p.join()
    #
    #    return study.best_trials