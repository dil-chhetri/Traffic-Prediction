import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle

warnings.filterwarnings('ignore')

df = pd.read_csv('traffic.csv')
df.drop(['ID'], axis=1, inplace=True)
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_new = df.copy()
df_new['Year'] = df_new['DateTime'].dt.year
df_new['Month'] = df_new['DateTime'].dt.month
df_new['Date'] = df_new['DateTime'].dt.day
df_new['Hour'] = df_new['DateTime'].dt.hour
df_new['Day'] = df_new.DateTime.dt.strftime('%A')

plt.figure(figsize=(20, 20), dpi = 400)
plot = sns.lineplot(x = df['DateTime'], y = "Vehicles", data = df, hue="Junction", palette="rainbow")
plot.set_title('Traffic Flow at Junctions')
plot.set_xlabel('Date')
plot.set_ylabel('Number of Vehicles')
plt.savefig('traffic_flow.png')

plt.figure(figsize=(20, 20), dpi = 400)
sns.pairplot(df_new, hue="Junction", palette="rainbow")
plt.savefig('traffic_flow_pairplot.png')

plt.figure(figsize=(20,20), dpi=400)
sns.histplot(x='Month', y='Vehicles', data=df_new, bins=20)
plt.savefig('traffic_flow_monthwise.png')


new_features_added= ['Year', 'Month', 'Date', 'Hour', 'Day']

for feature in new_features_added:
    plt.figure(figsize=(10, 4), dpi = 400)
    sns.lineplot(x = df_new[feature], y = "Vehicles", data = df_new, hue="Junction", palette="rainbow")
    plt.title("Number of Vehicles vs "+ str(feature))
    plt.xlabel(feature)
    plt.ylabel('Number of Vehicles')
    plt.legend(loc='best')
    plt.savefig(str(feature)+'.png')

plt.figure(figsize=(20, 15), dpi = 400)
sns.countplot(x = df_new["Year"], hue="Junction", data = df_new, palette="rainbow")
plt.title("Count of Taffic on Junction by years")
plt.xlabel("Year")
plt.ylabel('Number of Vehicles')
plt.legend(loc='best')
plt.savefig('count_by_years.png')


values_vehs, counts_vehs = np.unique(df_new['Year'], return_counts=True)
print(values_vehs, counts_vehs)

val_2017, count_2017 = np.unique(df_new['Month'][df_new['Year'] == 2017], return_counts=True)
print("2017 Monthwise: ", val_2017, count_2017)

encoded_values = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

df_new['Day'] = df_new['Day'].map(encoded_values)

df_1 = df_new[df_new['Junction'] == 1]

df_2 = df_new[df_new['Junction'] == 2]
df_2.reset_index(drop=True, inplace=True)

df_3 = df_new[df_new['Junction'] == 3]
df_3.reset_index(drop=True, inplace=True)

df_4 = df_new[df_new['Junction'] == 4]
df_4.reset_index(drop = True, inplace = True)

plt.figure(figsize=(10, 5))
sns.lineplot(x = df_1['DateTime'], y = "Vehicles", data = df_1, hue="Day", palette="rainbow")
plt.title("Number of Vehicles vs DateTime at Junction 1")
plt.xlabel("DateTime")
plt.ylabel('Number of Vehicles')
plt.legend(loc='best')
plt.savefig('junction_1.png')

plt.figure(figsize=(10, 5))
sns.lineplot(x = df_2['DateTime'], y = "Vehicles", data = df_2, hue="Day", palette="rainbow")
plt.title("Number of Vehicles vs DateTime at Junction 2")
plt.xlabel("DateTime")
plt.ylabel('Number of Vehicles')
plt.legend(loc='best')
plt.savefig('junction_2.png')


plt.figure(figsize=(10, 5))
sns.lineplot(x = df_3['DateTime'], y = "Vehicles", data = df_3, hue="Day", palette="rainbow")
plt.title("Number of Vehicles vs DateTime at Junction 3")
plt.xlabel("DateTime")
plt.ylabel('Number of Vehicles')
plt.legend(loc='best')
plt.savefig('junction_3.png')


plt.figure(figsize=(10, 5))
sns.lineplot(x = df_4['DateTime'], y = "Vehicles", data = df_4, hue="Day", palette="rainbow")
plt.title("Number of Vehicles vs DateTime at Junction 4")
plt.xlabel("DateTime")
plt.ylabel('Number of Vehicles')
plt.legend(loc='best')
plt.savefig('junction_4.png')


X1, y1 = df_1.drop(['Vehicles', 'DateTime', 'Junction'], axis=1), df_1['Vehicles']
X2, y2 = df_2.drop(['Vehicles', 'DateTime', 'Junction'], axis=1), df_2['Vehicles']
X3, y3 = df_3.drop(['Vehicles', 'DateTime', 'Junction'], axis=1), df_3['Vehicles']
X4, y4 = df_4.drop(['Vehicles', 'DateTime', 'Junction'], axis=1), df_4['Vehicles']


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=42)


X_trains = [X1_train, X2_train, X3_train, X4_train]
X_tests = [X1_test, X2_test, X3_test, X4_test]
y_trains = [y1_train, y2_train, y3_train, y4_train]
y_tests = [y1_test, y2_test, y3_test, y4_test]

models = [DecisionTreeRegressor(max_depth=5, random_state=42,
            min_samples_leaf=5, min_samples_split=5),
            LinearRegression(),
            RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, min_samples_leaf=5, min_samples_split=5),]


def plot_prediction_vs_datetime(df, y_pred, y_test, title):
    plt.figure(figsize=(10, 5))
    sns.lineplot(x = [i for i in range(df.shape[0])], y = y_test, data = df, palette="rainbow", label="Actual")
    sns.lineplot(x = [i for i in range(df.shape[0])],y= y_pred, data = df, palette="rainbow", label="Predicted")
    plt.title(title)
    plt.xlabel("DateTime")
    plt.ylabel('Number of Vehicles')
    plt.legend(loc='best')


models_short = ["DecisionTreeRegressor", "LinearRegressor", "RandomForestRegressor",]

model_used_Junction_1 = []
scores_train_Junction_1 = []
scores_test_Junction_1 = []
rmse_Junction_1 = []

for i in range(len(models)):
    models[i].fit(X_trains[0], y_trains[0])
    preds_test = models[i].predict(X_tests[0])
    preds_complete = models[i].predict(X1)
    plot_prediction_vs_datetime(X_tests[0], preds_test, y_tests[0], 'Prediction vs Actual for Junction 1')
    train_score = models[i].score(X_trains[0], y_trains[0])
    test_score = models[i].score(X_tests[0], y_tests[0])
    print(f"Accuracy score (Junction 1) for the model {models_short[i]} on the train set: {train_score}")
    print(f"Accuracy score (Junction 1) for the model {models_short[i]} on the test set: {test_score}")
    print(f"RMSE score (Junction 1) for the model {models_short[i]} on the test set: {mean_squared_error(y_tests[0], preds_test)**(1/2)}")
    model_used_Junction_1.append(models[i])
    scores_train_Junction_1.append(train_score)
    scores_test_Junction_1.append(test_score)
    rmse_Junction_1.append(mean_squared_error(y_tests[0], preds_test)**0.5)

print("FOR JUNCTION 1\n")
for i in range(len(models_short)):
    print(f"{models_short[i]}\n   Train Score: {scores_train_Junction_1[i]} - Test Score: {scores_test_Junction_1[i]} - RMSE: {rmse_Junction_1[i]}\n")


models_used_Junctions_2 = []
scores_train_Junctions_2 = []
scores_test_Junctions_2 = []
rmse_Junctions_2 = []

for i in range(len(models)):
    models[i].fit(X_trains[1], y_trains[1])
    preds_test = models[i].predict(X_tests[1])
    preds_complete = models[i].predict(X2)
    plot_prediction_vs_datetime(X_tests[1], preds_test, y_tests[1], 'Prediction vs Actual for Junction 2')
    train_score = models[i].score(X_trains[1], y_trains[1])
    test_score = models[i].score(X_tests[1], y_tests[1])
    print(f"Accuracy score (Junction 2) for the model {models_short[i]} on the train set: {train_score}")
    print(f"Accuracy score (Junction 2) for the model {models_short[i]} on the test set: {test_score}")
    print(f"RMSE score (Junction 2) for the model {models_short[i]} on the test set: {mean_squared_error(y_tests[1], preds_test)**(1/2)}")
    models_used_Junctions_2.append(models[i])
    scores_train_Junctions_2.append(train_score)
    scores_test_Junctions_2.append(test_score)
    rmse_Junctions_2.append(mean_squared_error(y_tests[1], preds_test)**0.5)

print("FOR JUNCTION 2\n")
for i in range(len(models_short)):
    print(f"{models_short[i]}\n   Train Score: {scores_train_Junctions_2[i]} - Test Score: {scores_test_Junctions_2[i]} - RMSE: {rmse_Junctions_2[i]}\n")

models_used_Junctions_3 = []
scores_train_Junctions_3 = []
scores_test_Junctions_3 = []
rmse_Junctions_3 = []

for i in range(len(models)):
    models[i].fit(X_trains[2], y_trains[2])
    preds_test = models[i].predict(X_tests[2])
    preds_complete = models[i].predict(X3)
    plot_prediction_vs_datetime(X_tests[2], preds_test, y_tests[2], 'Prediction vs Actual for Junction 3')
    train_score = models[i].score(X_trains[2], y_trains[2])
    test_score = models[i].score(X_tests[2], y_tests[2])
    print(f"Accuracy score (Junction 3) for the model {models_short[i]} on the train set: {train_score}")
    print(f"Accuracy score (Junction 3) for the model {models_short[i]} on the test set: {test_score}")
    print(
        f"RMSE score (Junction 3) for the model {models_short[i]} on the test set: {mean_squared_error(y_tests[2], preds_test) ** (1 / 2)}")
    models_used_Junctions_3.append(models[i])
    scores_train_Junctions_3.append(train_score)
    scores_test_Junctions_3.append(test_score)
    rmse_Junctions_3.append(mean_squared_error(y_tests[2], preds_test) ** 0.5)

print("FOR JUNCTION 3\n")
for i in range(len(models_short)):
    print(f"{models_short[i]}\n   Train Score: {scores_train_Junctions_3[i]} - Test Score: {scores_test_Junctions_3[i]} - RMSE: {rmse_Junctions_3[i]}\n")



models_used_Junctions_4 = []
scores_train_Junctions_4 = []
scores_test_Junctions_4 = []
rmse_Junctions_4 = []

for i in range(len(models)):
    models[i].fit(X_trains[3], y_trains[3])
    preds_test = models[i].predict(X_tests[3])
    preds_complete = models[i].predict(X4)
    plot_prediction_vs_datetime(X_tests[3], preds_test, y_tests[3], 'Prediction vs Actual for Junction 4')
    train_score = models[i].score(X_trains[3], y_trains[3])
    test_score = models[i].score(X_tests[3], y_tests[3])
    print(f"Accuracy score (Junction 4) for the model {models_short[i]} on the train set: {train_score}")
    print(f"Accuracy score (Junction 4) for the model {models_short[i]} on the test set: {test_score}")
    print(f"RMSE score (Junction 4) for the model {models_short[i]} on the test set: {mean_squared_error(y_tests[3], preds_test)**(1/2)}")
    models_used_Junctions_4.append(models[i])
    scores_train_Junctions_4.append(train_score)
    scores_test_Junctions_4.append(test_score)
    rmse_Junctions_4.append(mean_squared_error(y_tests[3], preds_test)**0.5)

print("FOR JUNCTION 4\n")
for i in range(len(models_short)):
    print(f"{models_short[i]}\n   Train Score: {scores_train_Junctions_4[i]} - Test Score: {scores_test_Junctions_4[i]} - RMSE: {rmse_Junctions_4[i]}\n")

final_models = {"Junction 1": models[2], "Junction 2": models[2], "Junction 3": models[2], "Junction 4": models[2]}
final_model_1 = final_models["Junction 1"]
final_model_2 = final_models["Junction 2"]
final_model_3 = final_models["Junction 3"]
final_model_4 = final_models["Junction 4"]

final_model_1.fit(X_trains[0], y_trains[0])
final_model_2.fit(X_trains[1], y_trains[1])
final_model_3.fit(X_trains[2], y_trains[2])
final_model_4.fit(X_trains[3], y_trains[3])

data1 = {"model" : final_model_1, "le_day" : encoded_values}
data2 = {"model" : final_model_2, "le_day" : encoded_values}
data3 = {"model" : final_model_3, "le_day" : encoded_values}
data4 = {"model" : final_model_4, "le_day" : encoded_values}

with open('final_model_1.pkl', 'wb') as f1:
    pickle.dump(data1, f1)

with open('final_model_2.pkl', 'wb') as f2:
    pickle.dump(data2, f2)

with open('final_model_3.pkl', 'wb') as f3:
    pickle.dump(data3, f3)

with open('final_model_4.pkl', 'wb') as f4:
    pickle.dump(data4, f4)

