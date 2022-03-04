# BreastCancer
I will using the Breast Cancer Wisconsin dataset to train a
model to predict if a patient has cancer or not ( benign vs malignant)¶
Im going to define a CNN based algorithm to receive the vector of features which is available in this dataset as input to predict the results.
The dataset is available at: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
30 features are extracted from each image.
Class distribution: 357 benign, 212 malignant
# This algorithm is run on kaggle and made 97% accuracy on the testset.
# Here is the link to the kaggle:
# https://www.kaggle.com/mortezaheidari/breast-cancer-classification-with-cnn/edit
# SOME NOTES:
## Since input is vector of features: Im Using L1 regularization of factor 0.01 applied to the kernel matrix, the reason is that L1 is acting like feature optimization, and sparsing the data.
## Max pooling is also so effective to provide robust results.
