import pickle
from utils.reuse_results import *
from configuration import Configuration as cfg

if __name__ == "__main__":
    results_dir = cfg.log_dir + "test/" 
    experiment_name = cfg.experiment_name
    results, recon_errors = load_results(results_dir, experiment_name)
    result = results[0][1]
    labels = np.array([x[0] for x in result])
    scores = np.array([x[1] for x in result])
    recon_errors = recon_errors[0]

    percentage = 50

    # Repickle results
    if cfg.test_name is None:
        results_path = results_dir + 'result_p%d.pkl'%(percentage)
        results_path_rec = results_dir + 'result_p%d_rec.pkl'%(percentage)
    else:
        results_path = results_dir + 'result_%s_p%d.pkl'%(cfg.test_name,percentage)
        results_path_rec = results_dir + 'result_%s_p%d_rec.pkl'%(cfg.test_name,percentage)
    
    with open(results_path, 'wb') as output:
        pickle.dump(result, output)
    with open(results_path_rec, 'wb') as output:
        pickle.dump(recon_errors, output)
