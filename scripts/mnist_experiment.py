import train_AAE
fold = 0
inlier_classes = [0]
total_classes = 10
num_folds = 5

train_AAE.main(fold,inlier_classes,total_classes,num_folds,bdd100k=False)
