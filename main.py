import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("student-mat.csv",delimiter=";").select_dtypes(exclude=["object"])
model = LinearRegression()

X = data.drop("G3",axis=1)
y = data["G3"]

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

model.fit(train_X,train_y)

y_predict = model.predict(val_X)

mae = mean_absolute_error(val_y,y_predict)

print(f'MAE : {mae}')

comparison = pd.DataFrame({
    "Actual" : val_y,
    "Predicted" : y_predict
})

print(comparison.head())