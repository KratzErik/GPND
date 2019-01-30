import pickle
import os
from configuration import Configuration as cfg
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc as compute_auc
import numpy as np


# The functions in this file are used by "novelty_detector/main()" if the configuration
# option "nd_original_GPND" is set to False (default setting).

def load_results(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name):
    # Loads results corresponding to 'experiment_name', stored in 'test_dir' and returns
    # a list of results (GPND novelty scores and labels) and a list of reconstruction errors 
    # corresponding to the same experiment.

    # This function is called by export_results, to save nd results as pickle files for 
    # GPND scores and reconstruction loss separately.

    assert(os.path.exists(test_dir))
    
    # Find relevant files
    if cfg.test_name is None:
        files = [filename for filename in os.listdir(test_dir) if "result_p" in filename]
    else:
        files = [filename for filename in os.listdir(test_dir) if "result_%s_p"%cfg.test_name in filename]
    print("Files to read: ", files)
    results = []
    recon_error_list = []
    for filename in files:
        if cfg.test_name is None:
            percentage = int(filename.replace("result_p","").replace(".pkl",""))
        else:
            percentage = int(filename.replace("result_%s_p"%cfg.test_name,"").replace(".pkl",""))

        with open(test_dir+filename,'rb') as file:
            [result, recon_errors] = pickle.load(file)

        # Add to lists
        results.append((percentage,result))
        recon_error_list.append(np.array(recon_errors))

    return results, recon_error_list

def get_performance_metrics(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name):

    results, recon_errors = load_results(test_dir, experiment_name)
    output_str = []
    for y in results:
        y_output_str = ""
        percentage = y[0]
        result = y[1]
        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]
        y_output_str += "Percentage of outliers: %d\n"%percentage
        if cfg.auroc:
            AUROC = roc_auc_score(y_true, y_scores)
            y_output_str += "\tAUROC:\t%.5f\n"%AUROC

        if cfg.auprc:
            pr, rc, _ = precision_recall_curve(y_true, y_scores)
            AUPRC = compute_auc(rc,pr)
            y_output_str += "\tAUPRC:\t%.5f\n"%AUPRC
        output_str.append(y_output_str)

    return output_str

def export_scores(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name, dataset = cfg.dataset):

    # Loads novelty scores and labels saved by 'novelty_detector/main()' and exports them
    # as scores, labels for both reconstruction error scores and GPND scores. 
    # Labels are 1 for outliers/novelties and 0 for inliers/normal samples. 
    # Scores are higher for outliers.
    # pkl files are saved to cfg.export_dir, defined in 'configuration.py'

    if cfg.training_mode == "GPND_default":
        alg_name = "GPND_pX"
    elif cfg.training_mode.lower() == "autoencoder":
        alg_name = "GPND_reconerr"

    results, recon_errors = load_results(test_dir, experiment_name)
    result = results[0][1]
    labels = np.array([x[0] for x in result])
    scores = np.array([x[1] for x in result])
    recon_errors = recon_errors[0]
    print("Scores: ", type(scores))
    print("Labels", type(labels))
    print("Rec err", type(recon_errors))
    print(recon_errors.shape)
    def export_one_score_type(score_vector, score_name):
        if cfg.test_name is None:
            results_filepath = cfg.export_results_dir + '%s_%s.pkl'%(dataset,score_name)
            exp_name_file = cfg.export_results_dir + 'experiment_names/%s_%s.txt'%(dataset,score_name)
        else:
            results_filepath = cfg.export_results_dir + '%s_%s_%s.pkl'%(dataset,score_name,cfg.test_name)
            exp_name_file = cfg.export_results_dir + 'experiment_names/%s_%s_%s.txt'%(dataset,score_name, cfg.test_name)
        
        pickle.dump([score_vector,labels], open(results_filepath,'wb'))

        print("Exported results to '%s'"%(results_filepath))

        # Update data source dict with experiment name
        with open(exp_name_file, 'w') as f:
            f.write(experiment_name)

    export_one_score_type(scores,"GPND_pX")
    export_one_score_type(recon_errors, "GPND_reconerr")