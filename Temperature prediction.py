import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
data = pd.read_csv("energydata_complete.csv")
data = data[["RH_1","T1"]]
# print(data.columns)


X_train, X_test, Y_train, Y_test = train_test_split(data["RH_1"], data["T1"], test_size=0.2, random_state=42)




X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test).reset_index()


def graph():
    data.plot(x='RH_1', y='T1', style='o')
    plt.title("Humidite vs Temperature")
    plt.xlabel("Humidite")
    plt.ylabel("Temperature")
    plt.show()

def VisualDiff(A_P_Data):
    A_P_Data.head(50).plot(kind="bar", figsize=(12,6))
    # plt.show()

def get_current_accuracy():
    test_y = np.array(Y_test[["T1"]])

    predicted_y = regressor.predict(X_test)
    t, e = predicted_y, test_y
    s = len(predicted_y)
    return 1 - sum(
        [
            abs(t[i] - e[i]) / e[i]
            for i in range(s)
            if e[i] != 0]
    ) / s
def graph(X, Y, slope, intercept, Title):
    plt.scatter(X, Y)
    plt.plot(X, slope*X + intercept, color="red")
    plt.title(Title)
    plt.xlabel("Humidity")
    plt.ylabel("Temprature")
    # plt.show()

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print("intecept: ",regressor.intercept_)
print("slope: ",regressor.coef_)

y_pred = regressor.predict(X_train)

y_pred = pd.DataFrame(y_pred, columns=["T1"])
print("Accuracy: ", get_current_accuracy())
graph(X_train, Y_train, regressor.coef_, regressor.intercept_, "Figure de regression lineaire")

# print(y_pred)
# print(Y_test)
Y_train = pd.DataFrame(Y_train).reset_index()
df = pd.DataFrame({"Actual": Y_train["T1"], "Predicted": y_pred["T1"]})
# print(df.to_string())
VisualDiff(df)
plt.show()