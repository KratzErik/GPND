import pickle
import os
from configuration import Configuration as cfg
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc as compute_auc


def load_results(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name):

    assert(os.path.exists(test_dir))
    
    files = [filename for filename in os.listdir(test_dir) if experiment_name+"_result_p" in filename]
    
    results = []
    for filename in files: 
        percentage = int(filename.replace(experiment_name+"_result_p","").replace(".pkl",""))
        result = pickle.load(filename)
        results.append((percentage,result))
    
    return results

def get_performance_metrics(test_dir = cfg.log_dir + "test/", experiment_name = cfg.experiment_name):
    output_str = "Result for experiment:\n"
    output_str += "Inliers: " + cfg.outliers_name
    output_str += "Outliers" + cfg.inliers_name

    results = load_results(test_dir, experiment_name)

    for y in results:
        percentage = y[0]
        result = y[1]
        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]

        float_spec = "%.5f"
        if cfg.auroc:
            AUROC = roc_auc_score(y_true, y_scores)
            output_str += "\tAUROC:\t"+ float_spec +"\n"%AUROC

        if cfg.auprc:
            pr, rc, _ = precision_recall_curve(y_true, y_scores)
            AUPRC = compute_auc(rc,pr)
            output_str += "\tAUPRC:\t" + float_spec + "\n"%AUPRC

    return output_str
