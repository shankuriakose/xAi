# Interpreting Pima Diabetes Prediction with SHAP, Eli5, and LIME

This project demonstrates how to interpret machine learning predictions for the Pima Indians Diabetes dataset using three popular explainability libraries: **SHAP**, **Eli5**, and **LIME**. The goal is to provide insights into model predictions and help users understand the factors influencing diabetes prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview
- **main.py**: Trains a machine learning model on the Pima Indians Diabetes dataset and applies SHAP, Eli5, and LIME for model interpretation.
- **Interpreting Pima Diabetes Prediction with SHAP,Eli5,LIME.ipynb**: Jupyter notebook with step-by-step code, visualizations, and explanations.

## Dataset
- **pima-indians-diabetes.csv**: Contains medical data for Pima Indian women, used for binary classification (diabetes prediction).
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) or [Kaggele Datasets](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


## Features
- Model training and evaluation
- Feature importance visualization with SHAP
- Model explanation with Eli5
- Local explanation with LIME
- Output visualizations (e.g., `output.png`)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shankuriakose/xAi.git
   cd xAI
   ```
2. **Set up a virtual environment:**
   ```bash
   python3 -m venv xAI
   source xAI/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or use Pipenv:
   ```bash
   pipenv install
   pipenv shell
   ```

## Usage
- **Open Jupyter:**
  Just type following command in the terminal and Open `Interpreting Pima Diabetes Prediction with SHAP,Eli5,LIME.ipynb`
  ```bash
  jupyter notebook
  ```

- **Run the main script (optional):**
  ```bash
  python main.py
  ```
- **Explore the notebook:**
  Open `Interpreting Pima Diabetes Prediction with SHAP,Eli5,LIME.ipynb` in JupyterLab or Jupyter Notebook.
  ```bash
  jupyter notebook
  ```

## Results
- Model performance metrics are printed in the console.
- Visualizations and explanations are saved as images (e.g., `output.png`).
- Interactive explanations are available in the notebook.

## Project Structure
```
├── Interpreting Pima Diabetes Prediction with SHAP,Eli5,LIME.ipynb
├── main.py
├── output.png
├── pima-indians-diabetes.csv
├── requirements.txt / Pipfile
├── xAI/ (virtual environment)
```

## License
This project is licensed under the MIT License.
