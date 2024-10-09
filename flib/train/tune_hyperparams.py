import optuna
import inspect
from flib.train import Clients
import multiprocessing

class HyperparamTuner():
    def __init__(self, study_name, obj_fn, train_dfs, val_dfs, seed=42, device='cpu', n_workers=1, storage=None, client=None, params=None):
        self.study_name = study_name
        self.obj_fn = obj_fn
        self.train_dfs = train_dfs
        self.val_dfs = val_dfs
        self.seed = seed
        self.device = device
        self.n_workers = n_workers
        self.storage = storage
        self.client = client
        self.params = params
    
    def objective(self, trial: optuna.Trial):
        params = {}
        for param in self.params['search_space']:
            if isinstance(self.params['search_space'][param], list):
                params[param] = trial.suggest_categorical(param, self.params['search_space'][param])
            elif isinstance(self.params['search_space'][param], tuple):
                if type(self.params['search_space'][param][0]) == int:
                    params[param] = trial.suggest_int(param, self.params['search_space'][param][0], self.params['search_space'][param][1])
                elif type(self.params['search_space'][param][0]) == float:
                    params[param] = trial.suggest_float(param, self.params['search_space'][param][0], self.params['search_space'][param][1])
            elif isinstance(self.params['search_space'][param], dict):
                params[param] = trial.suggest_categorical(param, list(self.params['search_space'][param].keys()))
                params[param+'_params'] = {}
                for subparam in self.params['search_space'][param][params[param]]:
                    if isinstance(self.params['search_space'][param][params[param]][subparam], list):
                        params[param+'_params'][subparam] = trial.suggest_categorical(params[param]+'_'+subparam, self.params['search_space'][param][params[param]][subparam])
                    elif isinstance(self.params['search_space'][param][params[param]][subparam], tuple):
                        if type(self.params['search_space'][param][params[param]][subparam][0]) == int:
                            params[param+'_params'][subparam] = trial.suggest_int(params[param]+'_'+subparam, self.params['search_space'][param][params[param]][subparam][0], self.params['search_space'][param][params[param]][subparam][1])
                        elif type(self.params['search_space'][param][params[param]][subparam][0]) == float:
                            params[param+'_params'][subparam] = trial.suggest_float(params[param]+'_'+subparam, self.params['search_space'][param][params[param]][subparam][0], self.params['search_space'][param][params[param]][subparam][1])
                    else:
                        params[param+'_params'][subparam] = self.params['search_space'][param][params[param]][subparam]
            else:
                params[param] = self.params['search_space'][param]
        
        for param in self.params['default']:
            if param not in params:
                if isinstance(self.params['default'][param], dict):
                    params[param] = next(iter(self.params['default'][param]))
                    params[param+'_params'] = {}
                    for subparam in self.params['default'][param][params[param]]:
                        params[param+'_params'][subparam] = self.params['default'][param][params[param]][subparam]
                else:
                    params[param] = self.params['default'][param]
        
        results = self.obj_fn(seed=self.seed, train_dfs=self.train_dfs, val_dfs=self.val_dfs, n_workers=self.n_workers, device=self.device, client=self.client, **params)
        avg_loss = 0.0
        for client in results:
            round = max(results[client].keys())
            avg_loss += results[client][round]['val']['loss'] / len(results)
        return avg_loss
    
    # def optimize(self, n_trials=10):
    #     study = optuna.create_study(storage=self.storage, sampler=optuna.samplers.TPESampler(), study_name=self.study_name, direction='minimize', load_if_exists=True)
    #     study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
    #     return study.best_trials
        
    def optimize(self, n_trials=100, n_jobs=10):
        # Create the study with RDB storage for parallel processing
        study = optuna.create_study(storage=self.storage, 
                                    sampler=optuna.samplers.TPESampler(), 
                                    study_name=self.study_name, 
                                    direction='minimize', 
                                    load_if_exists=True,
                                    pruner = optuna.pruners.HyperbandPruner())
        
        
        def run_study():
            study.optimize(self.objective, n_trials=n_trials // n_jobs)

        processes = []
        for _ in range(n_jobs):
            p = multiprocessing.Process(target=run_study)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return study.best_trials