#Created by Athanasios Tassiadamis
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, recall_score, accuracy_score, precision_score
import numpy as np
from sklearn.metrics import roc_auc_score

# Import dataset
strokeData = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Drop irrelevant columns (user.id)
strokeData = strokeData.drop('id', axis=1)

# Binary encode gender, marriage, and residence type
strokeData['ever_married'] = strokeData['ever_married'].map({'Yes': 1, 'No': 0})
strokeData['gender'] = strokeData['gender'].map({'Male': 1, 'Female': 0})
strokeData['Residence_type'] = strokeData['Residence_type'].map({'Urban': 1, 'Rural': 0})

# One-hot-encode smoking status and work type
strokeData = pd.get_dummies(strokeData, columns=['smoking_status', 'work_type'])

# Fill in NaN for BMI
strokeData['bmi'] = strokeData['bmi'].fillna(strokeData['bmi'].median())

# Split into feature and output data
X = strokeData.drop('stroke', axis=1)
y = strokeData['stroke']

# Undersample the majority class
# Get indices of the minority class
minority_class = strokeData[strokeData['stroke'] == 1]
majority_class = strokeData[strokeData['stroke'] == 0]

# Randomly sample from the majority class
majority_class_undersampled = majority_class.sample(n=len(minority_class))

# Combine the undersampled majority class with the minority class
undersampled_data = pd.concat([majority_class_undersampled, minority_class])

# Shuffle the dataset
undersampled_data = undersampled_data.sample(frac=1).reset_index(drop=True)

# Split into features and target variable
X_undersampled = undersampled_data.drop('stroke', axis=1)
y_undersampled = undersampled_data['stroke']

# Initialize Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True)

# Initialize lists to collect metrics
recall_scores = []
accuracy_scores = []
classification_reports = []
area_under_curve = []
precision_scores = []

#Create Neural Net for testing
input_dimensions = 17
num_nodes_h1 = 64
num_nodes_h2 = 32
num_nodes_h3 = 16
num_nodes_output = 1
activation_h1 = 'relu'
activation_h2 = 'relu'
activation_h3 = 'relu'
activation_output = 'sigmoid'
def keras_model(num1, num2, num_3, num_out, a_h1, a_h2, a_h3, a_out, input_dim):
    model1 = Sequential()
    model1.add(Dense(num1, input_dim=input_dim, activation=a_h1))
    model1.add(Dense(num2, activation=a_h2))
    model1.add(Dense(num_3, activation=a_h3))
    model1.add(Dense(num_out, activation=a_out))

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model1

# Stratified K-Fold Cross Validation
for train_index, test_index in skf.split(X_undersampled, y_undersampled):
    X_train, X_test = X_undersampled.iloc[train_index], X_undersampled.iloc[test_index]
    y_train, y_test = y_undersampled.iloc[train_index], y_undersampled.iloc[test_index]

    # Train the model
    model = keras_model(num_nodes_h1, num_nodes_h2, num_nodes_h3, num_nodes_output,activation_h1, activation_h2, activation_h3, activation_output, input_dimensions)
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Evaluate and collect metrics
    precision_scores.append(precision_score(y_test, y_pred_binary))
    recall_scores.append(recall_score(y_test, y_pred_binary))
    accuracy_scores.append(accuracy_score(y_test, y_pred_binary))
    classification_reports.append(classification_report(y_test, y_pred_binary, zero_division=1))
    area_under_curve.append(roc_auc_score(y_test, y_pred_binary))

# Print average metrics
print("\nClassification Reports for each fold:")
for i, report in enumerate(classification_reports):
    print(f"Fold {i + 1}:\n{report}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average AUC: {np.mean(area_under_curve):.4f}")