import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler


# -----------------------------
# 1. Load Data
# -----------------------------
names = [
    "Num_of_Preg",
    "Glucose_Conc",
    "BP",
    "Skin_Thickness",
    "TwoHour_Insulin",
    "BMI",
    "DM_Pedigree",
    "Age",
    "Class",
]
data = pd.read_csv("pima-indians-diabetes.csv", names=names)

print("Shape of the data set: ", data.shape)


# -----------------------------
# 2. Data Cleaning
# -----------------------------
def data_cleaner(df):
    # Replace biologically impossible zeros with median values
    zero_cols = ["Glucose_Conc", "BP", "Skin_Thickness", "BMI", "TwoHour_Insulin"]
    for col in zero_cols:
        df[col] = df[col].replace(0, df[col].median())
    return df


df = data_cleaner(data)

# -----------------------------
# 3. Features & Labels
# -----------------------------
Xfeatures = df.iloc[:, 0:8]
Ylabels = df["Class"]

# Scale features
scaler = Scaler()
X = scaler.fit_transform(Xfeatures)
X = pd.DataFrame(X, columns=names[0:8])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Ylabels, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Logistic Regression Model
# -----------------------------
logit = LogisticRegression(
    solver="liblinear",  # more stable for small datasets
    max_iter=500,  # allow convergence
    penalty="l2",  # regularization
    C=1.0,
)
logit.fit(X_train, y_train)

print("Accuracy Score of Logistic Regression:", logit.score(X_test, y_test))
