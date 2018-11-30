# test roc_auc

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# Score is higher for inliers
n = 100
y_score = np.arange(n)
y_true = np.concatenate([np.zeros((n//2,), dtype = int), np.ones((n//2,), dtype = int)])
np.random.shuffle(y_true)

print("Scores: ", type(y_score))
print("True labels:", type(y_true))

print("Score is higher for inliers")
print("With inlier as positive:")
print("\tAUROC = ", roc_auc_score(y_true, y_score))

y_true = 1 - y_true # turn true to false, etc
y_score = - y_score
print("With outlier as positive:")
print("\tAUROC = ", roc_auc_score(y_true, y_score))

