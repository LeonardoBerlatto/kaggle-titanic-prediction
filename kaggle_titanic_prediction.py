# -*- coding: utf-8 -*-
## Importing libs

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly.graph_objects as go

"""## Exploring the Data"""

train_set = pd.read_csv('train.csv')

test_set = pd.read_csv('test.csv')
test_set['Survived'] = pd.read_csv('gender_submission.csv')['Survived']

dataset = train_set.append(test_set).set_index('PassengerId')
dataset['Survived'] = dataset['Survived'].apply(lambda x : 'Yes' if x==1 else 'No')

dataset.head()

dataset.info()

"""### Survivence percentege"""

plt.figure(figsize=(10,8))
passengers_by_survivence = dataset['Survived'].value_counts()
passengers_by_survivence = passengers_by_survivence.rename_axis('Survived').reset_index(name='Counts')

labels = ['No', 'Yes']
colors = [] # TODO: decidir cores

fig = px.pie(passengers_by_survivence, values='Counts', names=labels, title='Passergers Survivence Percentage', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer="colab")

"""### Age distribuition"""

plt.figure(figsize=(10,8))

fig = px.histogram(dataset, x='Age', title='Age Distribuition', histnorm='', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer="colab")

"""### Number of survivors distribuition per age"""

plt.figure(figsize=(10,8))

age_per_survivence = px.histogram(dataset, x='Age', title='Number of survivors distribuited per age', color='Survived', color_discrete_sequence=px.colors.qualitative.G10)
age_per_survivence.show(renderer="colab")

"""### Survivor count per sex"""

plt.figure(figsize=(10,8))

survivor_count_per_sex = px.histogram(dataset, x="Survived", color="Sex", barmode='group', labels={'total_bill':'total bill'}, color_discrete_sequence=px.colors.qualitative.G10)
survivor_count_per_sex.show(renderer="colab")

"""### Correlation Heatmap"""

numeric_columns = ['Age', 'Survived', 'SibSp', 'Parch', 'Fare']

correlation = train_set[numeric_columns].corr()

import plotly.graph_objects as go

corr_heatmap = go.Figure(data=go.Heatmap(
                   colorscale=px.colors.sequential.Greys,
                   z=correlation.values,
                   x=numeric_columns,
                   y=numeric_columns,
                   hoverongaps = False))
corr_heatmap.update_layout(title='Correlation Heatmap')
corr_heatmap.show(renderer='colab')

"""## Feature selection"""

features = ['Age', 'Sex']
dependent_variable = 'Survived'

## adicionar conjunto de teste
X = train_set[features].append(test_set[features])
y = train_set[dependent_variable].append(test_set[dependent_variable])

print(X)

print(y)

"""### Feature Engineering"""

X['Sex'] = X['Sex'].apply(lambda x : 1 if x=='male' else 0)

from sklearn.impute import SimpleImputer
age_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age_imputer.fit(X[['Age']])

X['Age'] = age_imputer.transform(X[['Age']])[:, 0]

X = X.iloc[:,:].values
y = y.iloc[:].values

"""### Train-test split"""

X_train = X[:891, :]
X_test = X[891:, :]

y_train = y[:891]
y_test = y[891:]

"""### *Feature Scaling*"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""## Model building"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

"""### Using Random Forest with tuning"""

random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, min_samples_leaf=1, min_samples_split=10,random_state = 0)
random_forest.fit(X_train, y_train)

random_forest_pred = random_forest.predict(X_test)

print(accuracy_score(y_test, random_forest_pred))
writePredictionResults(decision_tree_model,X_test,'random_forest_tuned_with_age_and_sex.csv')

"""### Using Decision Tree with tuning"""

def writePredictionResults(model, X_test, file_name):
  y_pred = model.predict(X_test)

  results = pd.Series(y_pred, index=test_set['PassengerId'], name='Survived')
  results.to_csv(file_name, header=True)

def train_model(height):
  model = DecisionTreeClassifier(criterion = 'entropy', max_depth = height, random_state = 0)
  model.fit(X_train, y_train)
  return model

for height in range(1, 21): # 1-20
  model = train_model(height)
  y_pred = model.predict(X_test)
  
  print('--------------------------------------------------------------\n')
  print(f'Depth - {height}\n')
  print("Precision: " + str(accuracy_score(y_test, y_pred)))

"""Best Parameters"""

decision_tree_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
decision_tree_model.fit(X_train, y_train)
decision_tree_pred = decision_tree_model.predict(X_test)

print(accuracy_score(y_test, decision_tree_pred))
writePredictionResults(decision_tree_model,X_test,'decision_tree_tuned_with_age_and_sex.csv')

"""## Evaluation

"""

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

plt.figure(figsize=(8,6))
sns.heatmap(matrix, fmt='')

plt.title('Confusion Matrix')