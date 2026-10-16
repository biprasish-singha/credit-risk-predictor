# Credit Risk Predictor

### 

### Project Overview



This project predicts the likelihood of loan default using Machine Learning. The objective is to help financial institutions identify high-risk applicants and make informed lending decisions.



The solution follows a complete end-to-end Machine Learning workflow including:



* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Building
* Hyperparameter Tuning
* Threshold Optimization
* SHAP Explainability
* FastAPI Deployment



#### Live Demo

Application: https://credit-risk-predictor-7iu9.onrender.com

API Docs: https://credit-risk-predictor-7iu9.onrender.com/docs



### Problem Statement



Loan defaults can result in significant financial losses for lending institutions. By analyzing applicant information such as income, employment history, credit history, loan amount, and loan characteristics, we can proactively identify applicants who are at higher risk of default.



### Dataset Features

* #### Applicant Information
1. person\_age
2. person\_income
3. person\_home\_ownership
4. person\_emp\_length



* #### Loan Information
1. loan\_intent
2. loan\_grade
3. loan\_amnt
4. loan\_int\_rate
5. loan\_percent\_income



* #### Credit History Information
1. cb\_person\_default\_on\_file
2. cb\_person\_cred\_hist\_length



* #### Target Variable
1. loan\_default\_status (0 = No Default, 1 = Default)



### Feature Engineering



The following new feature was created:

* employment\_stability\_ratio -> The ratio between a person's employment years and age



### Models Evaluated

* Logistic Regression
* XGBoost



### Final Model



The Logistic Regression model achieved the best balance between interpretability and predictive performance.



Key reasons for selecting Logistic Regression:

* Highest overall ROC-AUC
* Strong recall for default detection
* Consistent cross-validation performance
* Good generalization ability
* Easy deployment and explainability



### Model Explainability



SHAP (SHapley Additive Explanations) was used to understand feature contributions and identify the most influential variables affecting model predictions.



### Model Deployment



The trained model is exposed through a REST API using FastAPI and deployed using Render.



### Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* FastAPI
* Uvicorn
* GitHub
* Render



### Future Improvements

* Docker Containerization
* CI/CD Pipeline
* Monitoring and Logging
* Model Retraining Pipeline



### Author

Biprasish Singha

Aspiring Data Scientist

