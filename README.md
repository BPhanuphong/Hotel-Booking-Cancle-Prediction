# Traditional Machine Learning and Multi-Layer Perception (MLP) Performance comparison on Binary Classification Data Set Study

## Executive Summary:
-
-

## Introduction:
- The purpose of this project is to compare the performance of tuned models between traditional Machine Learning Models (using scikit-learn and Grid Search to tune) and deep learning in MLP model (using TensorFlow and tuned dense layer, epoch, batch size, batch normalization, regularization) on tabular dataset.
- Use Hotel Booking Demand data set to predict booking cancellations in binary classification. (0: cancel, 1: not cancel)
- Tranditional Model: Logistic Regression, Decision Tree, Random Forest, XGBoost Calssifiter.

## Data Pre-processing:
- **Categorical:** transform data using Label Encoder
- **Numerical:** transform data using scaling values 


- **Imbalance Handling:** Our case is a binary classification problem and we found that the proportion of predicted value 0:1 is around 63:37, therefore; we decided to apply re-sampling strategies to obtain a more balanced data distribution as an effective solution to the imbalance problem.
This data set is not quite large so oversampling is in our consideration.
In addition, an improvement on duplicating examples from the minority class from the over-sample technique is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be the very effective and most widely used approach to synthesizing new examples called the **Synthetic Minority Oversampling (SMOTE) Technique** was used in this case.

![img](https://user-images.githubusercontent.com/80414593/189488119-d97b049a-b100-4569-b82c-76d9fc345aed.png)


## Model Development and Tuning: 
We developed 4 traditional machine learning models and tuned hyperparameter with Grid Search to find to top candidate model the tuning parameter of each model and Model result are shown in the table below

![img](https://user-images.githubusercontent.com/113247700/189488234-69e9791f-4db0-4bdd-8eeb-df247e2318b0.jpg)


## Model Development and Tuning: 
We developed 4 traditional machine learning models and tuned hyperparameter with Grid Search to find to top candidate model the tuning parameter of each model and Model result are shown in the table below

## Model Performance Evaluation:
Selected Model : XGBoost

Best Hyperparameter : 'gamma': 0, 'learning_rate': 0.25, 'max_depth': 10, 'n_estimators': 1000 

Overfitting : No overfitting found.

# Deep Learning Part:
**Creating Network architecture:**
- Input Layer
- Hidden Layer compose of 5 hidden layers with node = 12, 48, 64, 48 and 12 respectively
- Batch Normalization compose of 5 batch normalization layers by using Relu as activation function 
- Output Layer used sigmoid as activation function

**Training:** 
- GPU 0: Tesla T4 (UUID: GPU-86b357fa-3ace-7631-5530-b119e8b7cc24)

**Deep Learning Result:**

![img](https://user-images.githubusercontent.com/80414593/189488489-d358c46f-a6eb-4af6-acf2-7736da8c5a29.png)

- Train vs Validation
 
![img](https://user-images.githubusercontent.com/80414593/189488530-95ba05fe-4575-4a57-bc71-81262c35d76a.png)











