from configuration import Configuration as cfg
import train_AAE
import novelty_detector as nd

fold = 0
inlier_classes = [0]
total_classes = 10
num_folds = 5

train_AAE.main(fold,inlier_classes,total_classes,num_folds, cfg = cfg)
nd.main(fold,inlier_classes,total_classes,num_folds, cfg = cfg)
