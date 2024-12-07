# ML-for-Stroke-Prediction
Intro: Worked with a team of 4 to perform analysis of the Kaggle Stroke Prediction Dataset using Random Forest, Decision Trees, Neural Networks, KNN, SVM, and GBM.

DataSet: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Libraries Used: Pandas, Scitkitlearn, Keras, Tensorflow, MatPlotLib, Seaborn, and NumPy

DataSet Description: The Kaggle stroke prediction dataset contains over 5 thousand samples with 11 total features (3 continuous) including age, BMI, average glucose level, and more. The output attribute is a binary column titled “stroke”, with 1 indicating the patient had a stroke, and 0 indicating they did not.

Problems Faced: Highly imbalanced dataset (95% non-stroke, 5% stroke), missing values, irrelevant features, and un-encoded categorical variables.

PreProcessing Techniques: One-hot Encoding, feature selection, under-sampling, k-fold cross validation, and nullity encoding.

My Best Performing Models: My Decision Tree model achieved a recall of roughly 96%, meaning out of all the samples that were of patients with strokes, the model was accurrately able to predict 96% of them. My highest accurracy model was Random Forest, achieving an accurracy of roughly 80%, which was around the cap for this imbalanced dataset.

