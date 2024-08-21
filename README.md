# Heart Failure Prediction Dataset - Data Science Approach: Classification

## Overview

This project involves exploring the Heart Failure Prediction Dataset available on Kaggle to enhance data science skills. The dataset was obtained from Kaggle and is used solely for educational and learning purposes. The goal of this project is to thoroughly analyze the dataset, apply various data science techniques, and showcase findings and capabilities.

**Important Disclaimer:** This project is intended for learning and educational purposes only. The analyses and models developed should not be used for predicting heart disease or making medical decisions. The results are for academic exploration and do not reflect the accuracy or reliability needed for medical diagnoses.

**Citation:**  
fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction).

## Context

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs, and this dataset contains 11 features that can be used to predict possible heart disease.

Early detection and management of cardiovascular disease are crucial, and machine learning models can play a significant role in this process.

## Attribute Information

1. **Age**: Age of the patient (years)
2. **Sex**: Sex of the patient (M: Male, F: Female)
3. **ChestPainType**: Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
4. **RestingBP**: Resting blood pressure (mm Hg)
5. **Cholesterol**: Serum cholesterol (mm/dl)
6. **FastingBS**: Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
7. **RestingECG**: Resting electrocardiogram results (Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy)
8. **MaxHR**: Maximum heart rate achieved (Numeric value between 60 and 202)
9. **ExerciseAngina**: Exercise-induced angina (Y: Yes, N: No)
10. **Oldpeak**: Oldpeak = ST depression (Numeric value)
11. **ST_Slope**: The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
12. **HeartDisease**: Output class (1: heart disease, 0: Normal)

## Data Source

The dataset was created by combining different datasets previously available independently. It includes:

- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations
- Total: 1190 observations
- Duplicated: 272 observations

Final dataset: 918 observations

Each dataset used can be found under the Index of heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/).

## Table of Contents

- [Heart Failure Prediction Dataset - Data Science Approach: Classification](#heart-failure-prediction-dataset---data-science-approach-classification)
  - [Overview](#overview)
  - [Context](#context)
  - [Attribute Information](#attribute-information)
  - [Data Source](#data-source)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Objectives](#objectives)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Project Structure

- **data/**: Contains the raw dataset files.
- **notebooks/**: Jupyter notebooks for data exploration, cleaning, visualization, and running analyses and modeling.
- **model/**: Contains trained models and preprocessing objects.
  - `scaler.pkl`: A pickled StandardScaler object used for feature scaling.
  - `logistic_regression_model.pkl`: A pickled Logistic Regression model, the best-performing model.
- **new_data/**: Directory for new data and the notebook for processing it.
  - `process_new_data.ipynb`: Jupyter notebook to process new data using the trained model.
- **requirements.txt**: Lists the dependencies required to run the project.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/rdefays/heart-failure-prediction.git
    ```

2. **Navigate to the Project Directory:**
    ```bash
    cd heart-failure-prediction
    ```

3. **Install Dependencies:**
    Make sure you have Python and pip installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, follow these steps:

1. **Load the Dataset:**
    ```python
    import pandas as pd
    data = pd.read_csv('data/heart_failure_prediction.csv')
    ```

2. **Run Analysis:**
    Execute the Jupyter notebook or Python scripts to perform data analysis and modeling:
    ```bash
    jupyter notebook
    ```
    Or
    ```bash
    python scripts/analysis.py
    ```

3. **Load and Use Trained Models:**
    To use the trained models stored in the `model/` directory, follow these steps:

    ```python
    import joblib

    # Load the model and the scaler
    model = joblib.load('model/logistic_regression_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # Example usage
    # Prepare your data (make sure to apply the same scaling as during training)
    # X_test_scaled = scaler.transform(X_test)
    # predictions = model.predict(X_test_scaled)
    ```

4. **Process New Data:**
    To use the trained model on new data:
    - Place your new data file (e.g., `new_data.csv`) in the `new_data/` directory.
    - Use the `process_new_data.ipynb` notebook located in the same directory.
    - The notebook will:
      - Load the new data
      - Apply the same preprocessing steps as the training data
      - Use the trained model to make predictions
      - Output predictions and probabilities
    - The results will be saved in `new_data/predictions.csv`.

## Objectives

The primary objectives of this project are to:

- Explore and clean the dataset.
- Perform exploratory data analysis (EDA).
- Apply classification techniques to predict heart disease.
- Showcase the findings through visualizations and performance metrics.

## Results

Results and findings from the project are included in the Jupyter notebook or Python script. You can view visualizations, summaries, and any conclusions drawn from the analysis.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
