import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler as Scaler


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


def data_cleaner(df):
    # Replace 0 with Median not Mean
    df["BP"] = df["BP"].replace(to_replace=0, value=df["BP"].median())
    # Replace 0 with Median not Mean
    df["BMI"] = df["BMI"].replace(to_replace=0, value=df["BMI"].median())
    df["TwoHour_Insulin"] = df["TwoHour_Insulin"].fillna(df["TwoHour_Insulin"].median())
    # Replace 0 with Median not Mean
    df["Glucose_Conc"] = df["Glucose_Conc"].replace(
        to_replace=0, value=df["Glucose_Conc"].median()
    )
    df["Skin_Thickness"] = df["Skin_Thickness"].fillna(df["Skin_Thickness"].median())
    return df


df = data_cleaner(data)

Xfeatures = df.iloc[:, 0:8]
Ylabels = df["Class"]


scaler = Scaler()
X = scaler.fit_transform(Xfeatures)

X = pd.DataFrame(X, columns=names[0:8])

X_train, X_test, y_train, y_test = train_test_split(
    X, Ylabels, test_size=0.2, random_state=42
)


# Logit
logit = LogisticRegression()
logit.fit(X_train, y_train)


print("Accuracy Score of Logisitic::", logit.score(X_test, y_test))

# Prediction on A Single Sample
sample_pred = logit.predict(np.array(X_test.values[0]).reshape(1, -1))

print(sample_pred)
