from flask import Flask, request, render_template,redirect, url_for, session
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import mysql.connector

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'trafficpredictions'
}

app = Flask(__name__)
app.secret_key = 'dbc'


model_images = {
    'Random Forest Regression': {
        'junction_1': '/static/images/rfr_junc1.png',
        'junction_2': '/static/images/rfr_junc2.png',
        'junction_3': '/static/images/rfr_junc3.png',
        'junction_4': '/static/images/rfr_junc4.png'
    },
    'Linear Regression': {
        'junction_1': '/static/images/lr_junc1.png',
        'junction_2': '/static/images/lr_junc2.png',
        'junction_3': '/static/images/lr_junc3.png',
        'junction_4': '/static/images/lr_junc4.png'
    },
    'Decision Tree Regressor': {
        'junction_1': '/static/images/dtr_junc1.png',
        'junction_2': '/static/images/dtr_junc2.png',
        'junction_3': '/static/images/dtr_junc3.png',
        'junction_4': '/static/images/dtr_junc4.png'
    }
}


@app.route('/graphs', methods=['GET', 'POST'])
def show_graphs():

    selected_algorithm = 'Random Forest Regression'

    if request.method == 'POST':
        selected_algorithm = request.form.get('algorithm')


    plot_filename_junction1 = model_images[selected_algorithm]['junction_1']
    plot_filename_junction2 = model_images[selected_algorithm]['junction_2']
    plot_filename_junction3 = model_images[selected_algorithm]['junction_3']
    plot_filename_junction4 = model_images[selected_algorithm]['junction_4']

    algorithms = list(model_images.keys())

    return render_template(
        'demograph.html',
        algorithms=algorithms,
        selected_algorithm=selected_algorithm,
        plot_filename_junction1=plot_filename_junction1,
        plot_filename_junction2=plot_filename_junction2,
        plot_filename_junction3=plot_filename_junction3,
        plot_filename_junction4=plot_filename_junction4
    )


junction_data = {
        "Junction 1": {
            "DecisionTreeRegressor": {
                "Train Score": 0.8760,
                "Test Score": 0.8750,
                "RMSE": 8.06
            },
            "LinearRegressor": {
                "Train Score": 0.7069,
                "Test Score": 0.6984,
                "RMSE": 12.52
            },
            "RandomForestRegressor": {
                "Train Score": 0.8794,
                "Test Score": 0.8783,
                "RMSE": 7.95
            }
        },
        'Junction 2': {
            'DecisionTreeRegressor': {'Train Score': 0.8167, 'Test Score': 0.8023, 'RMSE': 3.24},
            'LinearRegressor': {'Train Score': 0.6067, 'Test Score': 0.5827, 'RMSE': 4.71},
            'RandomForestRegressor': {'Train Score': 0.8292, 'Test Score': 0.8142, 'RMSE': 3.14}
        },
        'Junction 3': {
            'DecisionTreeRegressor': {'Train Score': 0.3866, 'Test Score': 0.3535, 'RMSE': 8.27},
            'LinearRegressor': {'Train Score': 0.2468, 'Test Score': 0.2413, 'RMSE': 8.96},
            'RandomForestRegressor': {'Train Score': 0.4064, 'Test Score': 0.3828, 'RMSE': 8.08}
        },
        'Junction 4': {
            'DecisionTreeRegressor': {'Train Score': 0.5186, 'Test Score': 0.4185, 'RMSE': 2.79},
            'LinearRegressor': {'Train Score': 0.2225, 'Test Score': 0.2199, 'RMSE': 3.23},
            'RandomForestRegressor': {'Train Score': 0.5410, 'Test Score': 0.4501, 'RMSE': 2.71}
        }
    }


@app.route('/summary')
def summary():
    summary_data = {}


    for junction, models in junction_data.items():

        sorted_models = sorted(models.items(), key=lambda item: item[1]['RMSE'])


        summary_data[junction] = {
            "best": sorted_models[0],
            "middle": sorted_models[1],
            "least": sorted_models[2]
        }

    return render_template('demosummary.html', summary_data=summary_data)

with open('models/RF/final_model_1.pkl', 'rb') as f1:
    model1_data = pickle.load(f1)
    model1 = model1_data['model']

with open('models/RF/final_model_2.pkl', 'rb') as f2:
    model2_data = pickle.load(f2)
    model2 = model2_data['model']

with open('models/RF/final_model_3.pkl', 'rb') as f3:
    model3_data = pickle.load(f3)
    model3 = model3_data['model']

with open('models/RF/final_model_4.pkl', 'rb') as f4:
    model4_data = pickle.load(f4)
    model4 = model4_data['model']




with open('models/LR/linear_model_1.pkl', 'rb') as f1:
    model5_data = pickle.load(f1)
    model5 = model1_data['model']

with open('models/LR/linear_model_2.pkl', 'rb') as f2:
    model6_data = pickle.load(f2)
    model6 = model2_data['model']

with open('models/LR/linear_model_3.pkl', 'rb') as f3:
    model7_data = pickle.load(f3)
    model7 = model3_data['model']

with open('models/LR/linear_model_4.pkl', 'rb') as f4:
    model8_data = pickle.load(f4)
    model8 = model4_data['model']


with open('models/DTR/decision_model_1.pkl', 'rb') as f1:
    model9_data = pickle.load(f1)
    model9 = model1_data['model']

with open('models/DTR/decision_model_2.pkl', 'rb') as f2:
    model10_data = pickle.load(f2)
    model10 = model2_data['model']

with open('models/DTR/decision_model_3.pkl', 'rb') as f3:
    model11_data = pickle.load(f3)
    model11 = model3_data['model']

with open('models/DTR/decision_model_4.pkl', 'rb') as f4:
    model12_data = pickle.load(f4)
    model12 = model4_data['model']



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    session['prediction_made'] = False
    return render_template('predict.html')



@app.route('/predictions', methods=['POST'])
def predictions():
    model_algorithm = request.form.get('models')
    junction = request.form.get('junction')
    date = int(request.form.get('date'))
    month = int(request.form.get('month'))
    zone = int(request.form.get('junction'))
    day = float(request.form.get('day'))
    year = 2024


    time_str = request.form.get('time')
    time_parts = time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    current_time_decimal = hours + minutes / 60.0


    time_intervals = list(range(24))
    traffic_predictions = []
    traffic_conditions = []
    significant_changes = []
    traffic_condition_per_hour = {}


    light_traffic_count = 0
    normal_traffic_count = 0
    heavy_traffic_count = 0

    previous_condition = None
    for hour in time_intervals:
        time_decimal = hour + 0 / 60.0


        features = np.array([[date, month, zone, time_decimal, day]])


        if model_algorithm == '1':
            if junction == '1':
                prediction = model1.predict(features)
            elif junction == '2':
                prediction = model2.predict(features)
            elif junction == '3':
                prediction = model3.predict(features)
            else:
                prediction = model4.predict(features)
        elif model_algorithm == '2':
            if junction == '1':
                prediction = model5.predict(features)
            elif junction == '2':
                prediction = model6.predict(features)
            elif junction == '3':
                prediction = model7.predict(features)
            else:
                prediction = model8.predict(features)
        elif model_algorithm == '3':
            if junction == '1':
                prediction = model9.predict(features)
            elif junction == '2':
                prediction = model10.predict(features)
            elif junction == '3':
                prediction = model11.predict(features)
            else:
                prediction = model12.predict(features)

        predicted_value = prediction[0]
        traffic_predictions.append(predicted_value)


        if predicted_value > 7:
            current_condition = "Heavy Traffic"
            heavy_traffic_count += 1
        elif 5 <= predicted_value <= 7:
            current_condition = "Normal Traffic"
            normal_traffic_count += 1
        else:
            current_condition = "Light Traffic"
            light_traffic_count += 1

        traffic_conditions.append(current_condition)
        traffic_condition_per_hour[hour] = current_condition


        if previous_condition and current_condition != previous_condition:
            significant_changes.append((hour, current_condition))

        previous_condition = current_condition


    if traffic_conditions[-1] != traffic_conditions[0]:
        significant_changes.append((23, traffic_conditions[-1]))


    total_traffic = sum(traffic_predictions)


    average_traffic = total_traffic / len(time_intervals)


    current_features = np.array([[date, month, zone, current_time_decimal, day]])
    if model_algorithm == '1':
        if junction == '1':
            current_prediction = model1.predict(current_features)
        elif junction == '2':
            current_prediction = model2.predict(current_features)
        elif junction == '3':
            current_prediction = model3.predict(current_features)
        else:
            current_prediction = model4.predict(current_features)

    elif model_algorithm == '2':
        if junction == '1':
            current_prediction = model5.predict(current_features)
        elif junction == '2':
            current_prediction = model6.predict(current_features)
        elif junction == '3':
            current_prediction = model7.predict(current_features)
        else:
            current_prediction = model8.predict(current_features)

    elif model_algorithm == '3':
        if junction == '1':
            current_prediction = model9.predict(current_features)
        elif junction == '2':
            current_prediction = model10.predict(current_features)
        elif junction == '3':
            current_prediction = model11.predict(current_features)
        else:
            current_prediction = model12.predict(current_features)

    current_predicted_value = current_prediction[0]


    if current_predicted_value > 7:
        current_traffic_condition = "Heavy"
        traffic_gif_url = 'https://j.gifs.com/yEZ549.gif'
    elif 5 <= current_predicted_value <= 7:
        current_traffic_condition = "Normal"
        traffic_gif_url = 'https://i.pinimg.com/originals/89/a7/c1/89a7c103906549e860451f1b32ea0a0c.gif'
    else:
        current_traffic_condition = "Light"
        traffic_gif_url = 'https://cdn.secura.net/dims4/default/36a7cdd/2147483647/strip/true/crop/1024x410+0+0/resize/800x320!/quality/90/?url=https%3A%2F%2Fk2-prod-secura.s3.us-east-1.amazonaws.com%2Fbrightspot%2Fa0%2F56%2Fb14e48eebdd756e1bcfa861dace9%2Ffleet-intersection-main.jpg'


    if model_algorithm == '1':
        model = "RF"
    elif model_algorithm == '2':
        model = "LR"
    else:
        model = "DTR"


    junction_name = f"Junction {junction}"


    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, traffic_predictions, marker='o', linestyle='-', color='blue')


    for hour, condition in significant_changes:
        plt.axvline(x=hour, color='red', linestyle='--', label=f'Traffic change to {condition} at {hour}:00')

    plt.xticks(time_intervals)
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Predicted Traffic')
    plt.title(f'Predicted Traffic Throughout the Day with {model} at {junction_name}')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()


    bar_img = io.BytesIO()
    plt.figure(figsize=(8, 4))
    conditions = ['Light Traffic', 'Normal Traffic', 'Heavy Traffic']
    counts = [light_traffic_count, normal_traffic_count, heavy_traffic_count]
    plt.bar(conditions, counts, color=['green', 'orange', 'red'])
    plt.xlabel('Traffic Conditions')
    plt.ylabel('Occurrences')
    plt.title('Occurrences of Traffic Conditions Throughout the Day')
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bargraph_url = base64.b64encode(bar_img.getvalue()).decode()


    hourly_conditions_table = [
    {"time": f"{hour}:00", "condition": condition, "traffic": f"{traffic_predictions[hour]:.2f}"}
    for hour, condition in traffic_condition_per_hour.items()
]
    session['prediction_made'] = True

    db_connection = mysql.connector.connect(**db_config)
    cursor = db_connection.cursor()
    try:

        insert_query = """
                INSERT INTO predictions (date, time, model_type, prediction, total_traffic, average_traffic)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
        cursor.execute(insert_query, (
            f"{year}-{month}-{date}",
            time_str,
            model,
            current_predicted_value,
            total_traffic,
            average_traffic
        ))
        db_connection.commit()
    except Exception as e:
        print(f"Database insertion error: {e}")
    finally:
        cursor.close()
        db_connection.close()

    return render_template(
        'prediction.html',
        time_text = f'{time_str}',
        prediction_text=f'{current_predicted_value:.2f}',
        traffic_condition_text=f'{current_traffic_condition}',
        hourly_conditions_table=hourly_conditions_table,
        graph_url=f'data:image/png;base64,{graph_url}',
        bargraph_url=f'data:image/png;base64,{bargraph_url}',
        model_type=f'{model}',
        significant_changes_text=f'Significant traffic changes: {", ".join([f"{h}:00 to {c}" for h, c in significant_changes])}',
        traffic_gif_url = traffic_gif_url,
        total_traffic=f'{total_traffic:.2f}',
        average_traffic=f'{average_traffic:.2f}'
    )

@app.route('/predictions')
def preHome():

    if 'prediction_made' not in session or not session['prediction_made']:
        return redirect(url_for('predict'))
    return render_template('prediction.html')

@app.route("/delete", methods=["POST"])
def delete_data():
    row_id = request.form.get("id")
    print(f"Delete row: {row_id}")
    if row_id:
        db_connection = mysql.connector.connect(**db_config)
        cursor = db_connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s", (row_id,))
        db_connection.commit()
        db_connection.close()
    return redirect(url_for("history"))

@app.route('/history')
def history():
    db_connection = mysql.connector.connect(**db_config)
    cursor = db_connection.cursor()


    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    history_data = cursor.fetchall()
    print(history_data)
    cursor.close()
    db_connection.close()

    return render_template('history.html', history_data=history_data)



if __name__ == "__main__":
    app.run(debug=True)
