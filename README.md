# Predictive Maintenance for Manufacturing using XGBoost

This is a **Predictive Maintenance** application for a manufacturing setup that uses machine learning to predict potential failures and their types. The app is built using **Streamlit** for the user interface and employs **XGBoost** for both binary and multi-class classification tasks. It predicts whether a failure will occur and the type of failure based on machine parameters.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)

## Overview
This application allows users to input machine parameters and get real-time predictions for:
1. **Binary Classification**: Predict whether a machine failure will occur.
2. **Multi-class Classification**: Predict the type of failure (e.g., `Failure Type A`, `Failure Type B`, etc.).

The app is powered by **XGBoost** classifiers that have been pre-trained on historical data. It also features a user-friendly interface built using **Streamlit**.

## Dataset
The dataset used in this project includes machine sensor data, and it has been preprocessed to extract relevant features. The features include both **numerical** and **categorical** parameters related to machine performance.

## Models Used
The project uses two **XGBoost** models:
- **Binary classification model** for predicting if a failure will occur.
- **Multi-class classification model** for predicting the failure type.

## Features
- Input machine parameters interactively.
- Predict potential machine failures in real-time.
- View the type of failure, if predicted.
- Feature importance plots for both models.

## Installation
To run the application locally, you'll need to clone the repository and install the required dependencies.

### Prerequisites
Ensure that you have Python 3.7+ installed on your system.

### Steps to Set Up the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/predictive-maintenance-app.git
    cd predictive-maintenance-app
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure your pre-trained models (`xgb_target_model.pkl` and `xgb_failure_type_model.pkl`) are in the project directory. If not, you can train your own models using the included code.

## Usage
1. Run the Streamlit app locally by executing the following command:
    ```bash
    streamlit run app.py
    ```

2. The app will open in your browser. Enter the machine parameters, select the categorical feature (if applicable), and click **Predict** to view the predictions.

### Input Example:
- Parameter 1: 0.5
- Parameter 2: 2.1
- Parameter 3: 0.75
- Parameter 4: 1.0
- Parameter 5: 1.5
- Parameter 6: 0.9
- Categorical feature: Select from available options (e.g., `Machine A`, `Machine B`)

## Deployment
To deploy this app to a platform like **Heroku** or **Streamlit Cloud**, follow their respective documentation. For example, in Streamlit Cloud, you can simply connect your GitHub repo and the app will be deployed automatically.

Ensure your `requirements.txt` file is correctly set up and all model files are available in the repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The **XGBoost** library for its powerful gradient-boosting implementation.
- **Streamlit** for making data apps easy to build and deploy.
- **scikit-learn** for its preprocessing tools and easy-to-use ML pipelines.
