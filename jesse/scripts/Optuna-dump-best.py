# Dump best parameters from Optuna db
import optuna
import statistics
import hashlib
import json

study = optuna.create_study(study_name="Band5min", directions=["maximize", "maximize"],
                                storage="postgresql://optuna_user:optuna_password@localhost/optuna_db_10_2", load_if_exists=True)

def print_best_params():
    print("Number of finished trials: ", len(study.trials))

    trials = study.trials  # sorted(study.best_trials, key=lambda t: t.values)
    
    results = []
    parameter_list = []  # to eliminate redundant trials with same parameters
    candidates = {}
    score_treshold = 1  # 1.5
    std_dev_treshold = 5  # 3
    from jesse.routes import router
    import jesse.helpers as jh
    r = router.routes[0]
    StrategyClass = jh.get_strategy_class(r.strategy_name)
    r.strategy = StrategyClass()
        
    for trial in trials:
        if any(v < -1 for v in trial.values):  # 1
            continue
    
        mean_value = round(statistics.mean((*trial.values, trial.user_attrs['sharpe3'])), 3)
        std_dev = round(statistics.stdev((*trial.values, trial.user_attrs['sharpe3'])), 5)
        
        rounded_params = trial.params  # {key : round(trial.params[key], 5) for key in trial.params}
        # Inject payload HP to route
    
        hp_new = {}
        
        # Sort hyperparameters as defined in the strategy
        for p in r.strategy.hyperparameters():
            hp_new[p['name']] = rounded_params[p['name']]

        rounded_params = hp_new
        
        result_line = [trial.number, *trial.values, trial.user_attrs['sharpe3'],
                       trial.user_attrs['trades1'], trial.user_attrs['trades2'], trial.user_attrs['trades3'],
                       trial.user_attrs['fees1'], trial.user_attrs['fees2'], trial.user_attrs['fees3'],
                       trial.user_attrs['wr1'], trial.user_attrs['wr2'], trial.user_attrs['wr3'],
                       mean_value, std_dev, rounded_params]
        
        # Remove duplicates
        if trial.params not in parameter_list:
            results.append(result_line)
            parameter_list.append(trial.params)

            # If parameters meet criteria, add to candidates
            if mean_value > score_treshold and std_dev < std_dev_treshold:
                longest_param = 0
                
                for v in rounded_params.values():
                    if len(str(v)) > longest_param:
                        longest_param = len(str(v))
                
                # print("longest_param", longest_param)
                hash = ''.join([f'{value:0>{longest_param}}' for key, value in rounded_params.items()])
                hash = f'{hash}{longest_param}'
                # hash = parameters_to_emoji(rounded_params)  # 😄
                # candidates.append([hash, rounded_params])
                candidates[hash] = rounded_params
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    # for r in sorted_results:
    #     print(r)
    print(len(results))
    
    import csv
    
    # field names 
    fields = ['Trial #', 'Score1', 'Score2', 'Score3',
              'Trades1', 'Trades2', 'Trades3',
              'Fees1', 'Fees2', 'Fees3',
              'Winrate1', 'Winrate2', 'Winrate3',
              'Average', 'Deviation',
              'Parameters'] 
        
    with open('Results_10_5.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f, delimiter='\t', lineterminator='\n')
        
        write.writerow(fields)
        write.writerows(results)
        
    with open('10_5.py', 'w') as f:
        f.write("hps = ")
        f.write(json.dumps(candidates))
        # f.write("dnas = [\n")
        
        # for sr in candidates:
        #     f.write(f"{sr},\n")
        # f.write("}\n")
        
print_best_params()

# df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

# print("Best params: ", study.best_params)
# print("Best value: ", study.best_value)
# print("Best Trial: ", study.best_trial)
# print("Trials: ", study.trials)