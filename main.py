import pandas as pd 

data = pd.read_csv("student-mat.csv",delimiter=";")
print(data.isnull().sum())