# Traditional Machine Learning and Multi-Layer Perception (MLP) Performance comparison on Binary Classification Data Set Study

## Executive Summary:
-
-

## Introduction:
- The purpose of this project is to compare the performance of tuned models between traditional Machine Learning Models (using scikit-learn and Grid Search to tune) and deep learning in MLP model (using TensorFlow and tuned dense layer, epoch, batch size, batch normalization, regularization) on tabular dataset.
- Use Hotel Booking Demand data set to predict booking cancellations in binary classification. (0: cancel, 1: not cancel)
- Tranditional Model: Logistic Regression, Decision Tree, Random Forest, XGBoost Calssifiter.

## Assumption: 
- Based on our prior knowledge, traditional ML models might perform binary classification better than deep learning models on tabular datasets. Due to the complexities of deep learning, the model will take much more time to configure than traditional ML.


## Data Set Overview:
- **Task:** Binary Classification 
- **Objective:** To predict hotel cancelation transaction (Cancelled Yes or No)
- **Data Set:** Hotel Booking Demand
- **Data source:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?datasetId=511638&sortBy=voteCount&searchQuery=keras
- **Data shape:** 119,390 records with 32 Columns
- **Predicted column:** is_Canceled
- **Data Type:**

![image](https://user-images.githubusercontent.com/80414593/189491105-8c81c0af-9238-42de-a65d-2c1490786625.png)


## Data Exploration and Engineering Part:
### Exploratory Data Analysis:
- Bar chart to see imbalanced data.
- Box plot and Histogram to see

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Outlier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Distribution

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Skewness

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Mean, Median, Min, Max

- Correlation Matrix to the correlation between attributes.

### Data Validation:
- Data Imputation: We found that there are only 4 columns that have a null value which are company, agent, country, and children
- Handle missing values : 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Company has 94% Null so we drop this column.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Agent and country columns were replaced by mode.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Children column was replaced by mean.

### Feature Selection & Engineering:
- Create new 2 columns

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. total_members: sum of adults, babies, and children

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. total_stay: sum of stays_in_weekend_nights and stays_in_week_nights

- Drop 10 columns: 

![image](https://user-images.githubusercontent.com/80414593/189490182-7968566e-e0ff-4667-b854-e4203b5ade46.png)

### Data Pre-processing:
- **Categorical:** Transform data using Label Encoder
- **Numerical:** Transform data using scaling values 


### Imbalance Handling:
- Our case is a binary classification problem and we found that the proportion of predicted value 0:1 is around 63:37, therefore; we decided to apply re-sampling strategies to obtain a more balanced data distribution as an effective solution to the imbalance problem.
- This data set is not quite large so oversampling is in our consideration.
- In addition, an improvement on duplicating examples from the minority class from the over-sample technique is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be the very effective and most widely used approach to synthesizing new examples called the **Synthetic Minority Oversampling (SMOTE) Technique** was used in this case.

## Traditional Machine Learning Part 
### Data Splitting (Train/Validate/Test):

![image](https://user-images.githubusercontent.com/80414593/189491748-a7055280-30fe-418e-81e3-2989f6dda46b.png)

### Model Development and Tuning: 
- We developed 4 traditional machine learning models and tuned hyperparameter with Grid Search to find to top candidate model the tuning parameter of each model and Model result are shown in the table below

![image](https://user-images.githubusercontent.com/80414593/189491432-51161c2d-260e-4d90-82e6-be8af0bdfe65.png)

### Model Performance Evaluation:
- Selected Model : XGBoost
- Best Hyperparameter : 'gamma': 0, 'learning_rate': 0.25, 'max_depth': 10, 'n_estimators': 1000 
- Overfitting : No overfitting found.

![image](https://user-images.githubusercontent.com/80414593/189491451-c0e70fc4-94ce-4052-870c-3b4f8fae515c.png)

![img](https://user-images.githubusercontent.com/113247700/189488234-69e9791f-4db0-4bdd-8eeb-df247e2318b0.jpg)


# Deep Learning Part:
#### Creating Network architecture:

![image](https://user-images.githubusercontent.com/80414593/189491006-6c174e43-6d1e-41e5-b98a-39de4653263b.png)


- In each dataset, the number of input values is equal to the number of features which is 23  and there is only one output value for classification. 
- Hidden Layer compose of 5 hidden layers with node = 12, 48, 64, 48, and 12 respectively 
- Batch Normalization compose of 5 batch normalization layers by using Relu as the activation function for each layer.
- Dropout is not applied.
- Output Layer applied sigmoid as activation function

#### Optimizer:
- We use default optimizer ‘Adam’ to tune for best batch size and best epoch which are 200 and 200, respectively. 
#### Training: 
- Single loss and accuracy score strategy was applied to find the best model with a learning rate of 0.001.
- GPU 0: Tesla T4 (UUID: GPU-86b357fa-3ace-7631-5530-b119e8b7cc24)

### Deep Learning Result:
- The results of the deep learning model are shown in the table below and overfitting is not found.

![img](https://user-images.githubusercontent.com/80414593/189488489-d358c46f-a6eb-4af6-acf2-7736da8c5a29.png)



**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train vs Validation**
 
![img](https://user-images.githubusercontent.com/80414593/189488530-95ba05fe-4575-4a57-bc71-81262c35d76a.png)


# The Result of the Study Part: 

![image](https://user-images.githubusercontent.com/80414593/189490838-d126bc3b-ca54-452e-9fe2-b34a5d90a3bf.png)

- For our case study , traditional  machine learning model perform better than deep learning model in term of 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Training time of all 3 data set training,validation and test around 75 %

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - AUC Score  for all 3 data set which 3% for Train set and the Test set less than 0.00% which no significant.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - F1 score of class 1 and 0 for Train and Validation set around 1% - 3%

- While the deep learning model performs better in 
Inference Time of all 3 data sets training, validation, and test above 100%


# Discussion and Conclusion Part :

### Discussion:

### Conclusion:

## Appendix : 
### References:
https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e

### Group Member:

![image](https://user-images.githubusercontent.com/80414593/189489261-fd989ca7-73e2-4a9a-b1f8-27d1acd636a4.png)

This project is a part of the 

Subject: DADS7202 Deep Learning

Course: Data Analytics and Data Science 

Institution: National Institute of Development Administration (NIDA)







