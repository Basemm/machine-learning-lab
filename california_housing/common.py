import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
  return pd.read_csv("./data/california_housing.csv")

def get_dataset():
  df = load_data()

  df["income_category"] = pd.cut(
      df["median_income"],
      bins=[0., 1.5, 3.0, 4.5, 6., float("inf")],
      labels=[1, 2, 3, 4, 5]
  )

  stratified_train_set, stratified_test_set = train_test_split(df, test_size=0.2, stratify=df["income_category"], random_state=42)

  X_train = stratified_train_set.drop("median_house_value", axis=1)
  y_train = stratified_train_set["median_house_value"]

  X_test = stratified_test_set.drop("median_house_value", axis=1)
  y_test = stratified_test_set["median_house_value"]

  return X_train, y_train, X_test, y_test