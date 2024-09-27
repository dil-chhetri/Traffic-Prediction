from flask import Flask, request, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)

# Load all the models for the four junctions
# Assuming models were saved as dictionaries
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



#Linear Regression Algorithm
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

# Decion Tree Regression
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


# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    model_algorithm = request.form.get('models')
    junction = request.form.get('junction')
    temperature = float(request.form.get('temperature'))
    weather = float(request.form.get('weather'))
    zone = int(request.form.get('zone'))
    day = float(request.form.get('day'))

    # Convert current time to decimal hours
    time_str = request.form.get('time')
    time_parts = time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    current_time_decimal = hours + minutes / 60.0

    # Initialize lists to store time, traffic predictions, and traffic conditions
    time_intervals = list(range(24))  # From 0:00 to 23:00
    traffic_predictions = []
    traffic_conditions = []
    significant_changes = []  # To store times of significant traffic changes
    traffic_condition_per_hour = {}  # To store condition for each hour

    previous_condition = None
    for hour in time_intervals:
        time_decimal = hour + 0 / 60.0  # No minutes, just the hour as decimal

        # Features for each hour prediction
        features = np.array([[temperature, weather, zone, time_decimal, day]])

        # Select the model based on the algorithm and junction
        if model_algorithm == '1':  # Random Forest
            if junction == '1':
                prediction = model1.predict(features)
            elif junction == '2':
                prediction = model2.predict(features)
            elif junction == '3':
                prediction = model3.predict(features)
            else:
                prediction = model4.predict(features)
        elif model_algorithm == '2':  # Linear Regression
            if junction == '1':
                prediction = model5.predict(features)
            elif junction == '2':
                prediction = model6.predict(features)
            elif junction == '3':
                prediction = model7.predict(features)
            else:
                prediction = model8.predict(features)
        elif model_algorithm == '3':  # Decision Tree
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

        # Determine traffic condition for this hour
        if predicted_value > 15:
            current_condition = "Heavy Traffic"
        elif 5 <= predicted_value <= 15:
            current_condition = "Normal Traffic"
        else:
            current_condition = "Light Traffic"

        traffic_conditions.append(current_condition)
        traffic_condition_per_hour[hour] = current_condition  # Store condition for this hour

        # Track significant changes in traffic condition
        if previous_condition and current_condition != previous_condition:
            significant_changes.append((hour, current_condition))

        previous_condition = current_condition

    # Handle the edge case of the last hour (23:00) to midnight transition
    if traffic_conditions[-1] != traffic_conditions[0]:  # Check if last hour is different from start
        significant_changes.append((23, traffic_conditions[-1]))

    # Get prediction for the current time (input by the user)
    current_features = np.array([[temperature, weather, zone, current_time_decimal, day]])
    if model_algorithm == '1':  # Random Forest
        if junction == '1':
            current_prediction = model1.predict(current_features)
        elif junction == '2':
            current_prediction = model2.predict(current_features)
        elif junction == '3':
            current_prediction = model3.predict(current_features)
        else:
            current_prediction = model4.predict(current_features)
    elif model_algorithm == '2':  # Linear Regression
        if junction == '1':
            current_prediction = model5.predict(current_features)
        elif junction == '2':
            current_prediction = model6.predict(current_features)
        elif junction == '3':
            current_prediction = model7.predict(current_features)
        else:
            current_prediction = model8.predict(current_features)
    elif model_algorithm == '3':  # Decision Tree
        if junction == '1':
            current_prediction = model9.predict(current_features)
        elif junction == '2':
            current_prediction = model10.predict(current_features)
        elif junction == '3':
            current_prediction = model11.predict(current_features)
        else:
            current_prediction = model12.predict(current_features)

    current_predicted_value = current_prediction[0]

    # Determine traffic condition for the current time
    if current_predicted_value > 15:
        current_traffic_condition = "Heavy Traffic"
    elif 5 <= current_predicted_value <= 15:
        current_traffic_condition = "Normal Traffic"
    else:
        current_traffic_condition = "Light Traffic"

    if model_algorithm == '1':
        model = "Random Forest Regressor"
    elif model_algorithm == '2':
        model = "Linear Regression"
    else:
        model = "Decision Tree Regressor"

    if junction == '1':
        junction_name = "Junction 1"
    elif junction == '2':
        junction_name = "Junction 2"
    elif junction == '3':
        junction_name = "Junction 3"
    elif junction == '4':
        junction_name = "Junction 4"

    # Generate the plot for traffic throughout the day with significant changes
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, traffic_predictions, marker='o', linestyle='-', color='blue')

    # Highlight the significant changes in traffic condition
    for hour, condition in significant_changes:
        plt.axvline(x=hour, color='red', linestyle='--', label=f'Traffic change to {condition} at {hour}:00')

    # Annotate traffic conditions at each hour on the graph
    # for hour, condition in traffic_condition_per_hour.items():
    #     plt.text(hour, traffic_predictions[hour] + 0.5, condition, ha='center', fontsize=8)

    plt.xticks(time_intervals)  # Show every hour on the x-axis
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Predicted Traffic')
    plt.title(f'Predicted Traffic Throughout the Day with Traffic Conditions with {model} at {junction_name}')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Render the HTML template with the current prediction, hourly traffic conditions, and graph
    hourly_conditions = ", ".join(
        [f"{hour}:00 - {condition}" for hour, condition in traffic_condition_per_hour.items()])

    return render_template(
        'index.html',
        prediction_text=f'Predicted Traffic at {time_str}: {current_predicted_value:.2f}',
        traffic_condition_text=f'Condition at {time_str}: {current_traffic_condition}',
        hourly_conditions_text=f'Hourly Traffic Conditions: {hourly_conditions}',
        graph_url=f'data:image/png;base64,{graph_url}',
        model_type=f'Model used for prediction: {model}',
        significant_changes_text=f'Significant traffic changes: {", ".join([f"{h}:00 to {c}" for h, c in significant_changes])}',
    )

if __name__ == "__main__":
    app.run(debug=True)
