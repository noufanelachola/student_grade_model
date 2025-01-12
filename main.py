import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("student-mat.csv",delimiter=";")

categorical_columns = data.select_dtypes(include="object").columns

binary_columns = [col for col in categorical_columns if data[col].nunique() == 2 ]
columns_to_drop = [col for col in categorical_columns if col not in binary_columns]

data.drop(columns_to_drop,axis=1,inplace=True)

# Binary_Encoding
# for col in binary_columns:
#     col_values = data[col].unique()
#     data[col] = data[col].map({col_values[0]:0, col_values[1]:1})

labelEncoder = LabelEncoder()
for col in binary_columns:
    data[col] = labelEncoder.fit_transform(data[col])

print(data.head())

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