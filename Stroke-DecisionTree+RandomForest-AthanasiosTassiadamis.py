# Created by Athanasios Tassiadamis
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score, precision_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
strokeData = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Drop irrelevant columns (user.id)
strokeData = strokeData.drop('id', axis=1)

#Visualize nullity of dataset

# Count non-null values in each column
not_null = strokeData.notnull().sum()

# Plot the bar graph
plt.figure(figsize = (11,9))
not_null.plot(kind='bar', color='pink', edgecolor='black')

# Customize the plot
plt.title('Valid Values per Column')
plt.xlabel('Columns')
plt.ylabel('Count of Valid Values')
plt.xticks(rotation=30, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=1)

# Show the plot
plt.show()

#Show the balance of the dataset
# Count the occurrences of each class
class_counts = strokeData['stroke'].value_counts()

# Plot the bar graph
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['pink', 'violet'], edgecolor='black')

# Customize the plot
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No-Stroke', 'Stroke'], fontsize=10, rotation=0)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "Feature" grouped by "Output"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='age',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='age',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of Age Grouped by Stroke')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "Feature" grouped by "Output"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='bmi',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='bmi',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of Bmi Grouped by Stroke')
plt.xlabel('Bmi')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "avg_glucose_level" grouped by "Stroke"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='avg_glucose_level',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='avg_glucose_level',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of avg_glucose_level Grouped by Stroke')
plt.xlabel('avg_glucose_level')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Binary encode gender, marriage, and residence type
strokeData['ever_married'] = strokeData['ever_married'].map({'Yes': 1, 'No': 0})
strokeData['gender'] = strokeData['gender'].map({'Male': 1, 'Female': 0})
strokeData['Residence_type'] = strokeData['Residence_type'].map({'Urban': 1, 'Rural': 0})

# One-hot-encode smoking status and work type
strokeData = pd.get_dummies(strokeData, columns=['smoking_status', 'work_type'])

# Fill in NaN for BMI
strokeData['bmi'] = strokeData['bmi'].fillna(strokeData['bmi'].median())

# Split into feature and output data
#Dropping bmi and avg_glucose_lvl could have positive effect
X = strokeData.drop(['stroke'], axis=1)
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
precision_scores = []
recall_scores = []
accuracy_scores = []
classification_reports = []
area_under_curve = []

# Stratified K-Fold Cross Validation
for train_index, test_index in skf.split(X_undersampled, y_undersampled):
    X_train, X_test = X_undersampled.iloc[train_index], X_undersampled.iloc[test_index]
    y_train, y_test = y_undersampled.iloc[train_index], y_undersampled.iloc[test_index]

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    #model = DecisionTreeClassifier(max_depth=2, min_samples_split=5, min_samples_leaf=1, criterion="entropy")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate and collect metrics
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    classification_reports.append(classification_report(y_test, y_pred, zero_division=1))
    area_under_curve.append(roc_auc_score(y_test, y_pred))

# Print average metrics
print("\nClassification Reports for each fold:")
for i, report in enumerate(classification_reports):
    print(f"Fold {i + 1}:\n{report}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average AUC: {np.mean(area_under_curve):.4f}")
