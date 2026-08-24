# ***

# ## Deep Learning & Generative AI - Exam Practice

# ## PRACTICAL 2 : MLP Activation and Layer Experiments

# ***

# #### Objective: This practical uses the **handwritten digits dataset**. Each sample is an $8 \times 8$ grayscale image representing one digit from 0 to 9.
#
# ##### Input: 64 numerical pixel-intensity features.
#
# ##### Output: 10 mutually exclusive classes.
#

# #### Theory Notes:
#
# * **MLP (Multi-Layer Perceptron)** is a feed-forward neural network with one or more hidden layers.
# * **Activation functions** introduce non-linearity and affect how the network learns.
# * Increasing the number of hidden layers/neurons increases model capacity.
# * In this practical, different **activation functions** and **hidden-layer configurations** are compared using validation accuracy.
# * The dataset is divided into training, validation and testing sets, and the input features are standardized before training.

# ***

# ### 1] import libraries

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# ### 2] load and inspect the data

# Load handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target

print('Input shape:', X.shape)
print('Target shape:', y.shape)
print('Number of classes:', len(np.unique(y)))
print('Classes:', np.unique(y))

# ### 3] train validation test split

# 80% training/validation and 20% final testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Split remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split( X_train_val, y_train_val, test_size=0.25,
                                                  random_state=42, stratify=y_train_val)

print('Training samples:', len(X_train))
print('Validation samples:', len(X_val))
print('Testing samples:', len(X_test))

# ### 4] feature standardization

scaler = StandardScaler()

# Fit only on training data
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print('Standardization completed.')

# ### 5] multi-layer perceptron architecture

# Different hidden-layer configurations are tested to study the effect of network depth and size.
#
# * **Small-Shallow:** `(32,)`
# * **Medium-Two-Layer:** `(64, 32)`
# * **Deep-Three-Layer:** `(128, 64, 32)`
#
# The output layer contains **10 classes**, corresponding to digits 0–9.

layer_configurations = {
    'Small-Shallow': (32,),
    'Medium-Two-Layer': (64, 32),
    'Deep-Three-Layer': (128, 64, 32)
}

activation_functions = ['relu', 'tanh', 'logistic']

# ### 6] experimental design

# Create all activation-function and architecture combinations
experiment_plan = []

for activation in activation_functions:
    for name, layers in layer_configurations.items():
        experiment_plan.append([activation, name, layers])

plan = pd.DataFrame(experiment_plan, columns=['Activation', 'Configuration', 'Hidden Layers'])

display(plan)

# ### 7] training

results = []
models = {}

for activation in activation_functions:
    for name, layers in layer_configurations.items():
        
        # Build MLP model
        model = MLPClassifier( hidden_layer_sizes=layers, activation=activation, solver='adam',
                              max_iter=300, early_stopping=True, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Validation prediction
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        
        models[f'{activation} - {name}'] = model
        results.append([activation, name, acc, f1])

results_df = pd.DataFrame( results, columns=['Activation', 'Configuration', 'Validation Accuracy', 'Validation F1'])

print('All experiments completed.')

# ### 8] experimental results

# Rank models by validation accuracy
results_df = results_df.sort_values(by=['Validation Accuracy', 'Validation F1'],
                                    ascending=False).reset_index(drop=True)

results_df.index = results_df.index + 1
results_df.index.name = 'Rank'

display(results_df)

# ***

# ### 10] quick revision
#
# * **MLP:** Feed-forward neural network with one or more hidden layers.
# * **ReLU:** Common non-linear activation; `max(0, x)`.
# * **Tanh:** Outputs values between -1 and 1.
# * **Logistic / Sigmoid:** Outputs values between 0 and 1.
# * **Standardization:** Scales input features before training.
# * **Validation accuracy:** Used to compare the different models.
# * **More layers/neurons:** Generally increase model capacity but may also increase training complexity.
# * **Best model:** The configuration with the highest validation accuracy in the results table.

# ***
