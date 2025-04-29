# Parkinson-s-Disease-Detection-using-Voice-Features

A machine learning model that predicts whether a person has Parkinson's Disease based on biomedical voice measurements.

# Data

This project utilizes the Oxford Parkinson's Disease Detection Dataset obtained from the UCI Machine Learning Repository.

This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.

# Model Training and Evaluation

The models trained and evaluated for this project are

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

SHAP and LIME has been utlized for model explainability.

# Application

A simple streamlit web application is created that allows users to detect parkinson's disease using the 22 input voice features utilized in this ML model. The user has the ability to enter the values of each feature and detect the disease presence.

Future Enhancements:

- Enabling users to chose one of the two options of manual data entry and csv file upload, for disease detection.
- Deployment of the application.

To execute the application locally, enter the following command in your terminal,

```bash
streamlit run app.py
```
