import pickle
import os
from configuration import Configuration as cfg
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc as compute_auc


def load_results(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name):

    assert(os.path.exists(test_dir))
    
    # Find relevant files
    if cfg.test_name is None:
        files = [filename for filename in os.listdir(test_dir) if "result_p" in filename]
    else:
        files = [filename for filename in os.listdir(test_dir) if "result_p" in filename and cfg.test_name in filename]
    results = []
    recon_error_list = []
    for filename in files:
        if cfg.test_name is None:
            percentage = int(filename.replace("result_p","").replace(".pkl",""))
        else:
            percentage = int(filename.replace("result_%s_p"%cfg.test_name,"").replace(".pkl",""))

        with open(test_dir+filename,'rb') as file:
            [result, recon_errors] = pickle.load(file)
        results.append((percentage,result))
        recon_error_list.append(recon_errors)

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

    if cfg.training_mode == "GPND_default":
        alg_name = "GPND_pX"
    elif cfg.training_mode.lower() == "autoencoder":
        alg_name = "GPND_reconerr"

    results, recon_errors = load_results(test_dir, experiment_name)
    print("results: ", results)
    result = results[0][1]
    labels = [x[0] for x in result]
    scores = [x[1] for x in result]

    def export_one_score_type(score_vector, score_name):
        if Cfg.test_name is None:
            results_filepath = '/home/exjobb_resultat/data/%s_%s.pkl'%(dataset,score_name)
            exp_name_file = '/home/exjobb_resultat/data/experiment_names/%s_%s.txt'%(dataset,score_name)
        else:
            results_filepath = '/home/exjobb_resultat/data/%s_%s_%s.pkl'%(dataset,score_name,cfg.test_name)
            exp_name_file = '/home/exjobb_resultat/data/experiment_names/%s_%s_%s.txt'%(dataset,score_name, cfg.test_name)
        
        pickle.dump([score_vector,labels], open(results_filepath,'wb'))

        print("Exported results to '%s'"%(results_filepath))

        # Update data source dict with experiment name
        with open(exp_name_file, 'w') as f:
            f.write(experiment_name)

    export_one_score_type(scores,"GPND_pX")
    export_one_score_type(recon_errors[0], "GPND_reconerr")

    # common_results_dict = pickle.load(open('/home/exjobb_resultat/data/name_dict.pkl','rb'))
    # common_results_dict[dataset][alg_name] = experiment_name
    # pickle.dump(common_results_dict,open('/home/exjobb_resultat/data/name_dict.pkl','wb'), protocol=2)
    # print("Updated entry ['%s']['%s'] = '%s' in file /home/exjobb_resultat/data/name_dict.pkl"%(dataset,alg_name,experiment_name))
