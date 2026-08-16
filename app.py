import os
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib

app = Flask(__name__)
MODEL_FILE = 'railway_model_1cr.pkl'

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found! Run 'python script.py' first.")

HIGH_DEMAND_CODES = {
    'NDLS', 'ANVT', 'DLI', 'NZM', 'PNBE', 'BJU', 'DNR', 'CSMT', 'LTT', 'BCT', 'BDTS', 
    'HWH', 'SDAH', 'KOAA', 'MAS', 'MS', 'SBC', 'YPR', 'BSB', 'DDU', 'GKP', 'LKO', 
    'CNB', 'PRYJ', 'ADI', 'ST', 'BRC', 'PUNE', 'NGP', 'BPL', 'JP', 'AGC', 'BBS', 
    'VSKP', 'BZA', 'SC', 'HYB', 'GHY', 'CDG', 'ASR', 'LDH'
}

def evaluate_route_tier(src_code, dest_code):
    src_code, dest_code = src_code.upper().strip(), dest_code.upper().strip()
    if src_code in HIGH_DEMAND_CODES or dest_code in HIGH_DEMAND_CODES:
        return 1
    return 2

def check_is_festival(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        if dt.month in [3, 10, 11]:
            return 1
    except:
        pass
    return 0

def calculate_days_left(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = (dt - datetime.today()).days
        return max(0, delta)
    except:
        return 10

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        src_code = data.get('source_station', '')
        dest_code = data.get('destination_station', '')
        journey_date = data.get('journey_date', datetime.today().strftime('%Y-%m-%d'))
        waitlist_num = int(data.get('waitlist_number', 10))
        coach_type = int(data.get('coach_type', 1))
        quota = int(data.get('quota', 0))
        train_type = 0 

        days_left = calculate_days_left(journey_date)

        # STRICT REAL-WORLD RULE: If traveling tomorrow (0-2 days out) with WL > 10, confirmation is practically impossible
        if days_left <= 2 and waitlist_num > 10:
            prob_rounded = 4.5
            status = "High Risk of Waitlist / Cancellation"
        else:
            route_tier = evaluate_route_tier(src_code, dest_code)
            is_festival = check_is_festival(journey_date)

            features = pd.DataFrame([[route_tier, days_left, waitlist_num, train_type, coach_type, quota, is_festival]],
                                    columns=['route_tier', 'days_left', 'waitlist_num', 'train_type', 'class', 'quota', 'is_festival'])
            
            prediction_prob = model.predict_proba(features)[0][1] * 100
            prob_rounded = round(float(prediction_prob), 2)
            
            status = "High Chance of Confirmation" if prob_rounded > 50 else "High Risk of Waitlist / Cancellation"

        return jsonify({
            'success': True,
            'probability': prob_rounded,
            'status': status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stations.json')
def serve_stations():
    return send_from_directory('.', 'stations.json', mimetype='application/json')

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('.', 'manifest.json', mimetype='application/manifest+json')

@app.route('/icon-512.png')
def icon_512():
    return send_from_directory('.', 'icon-512.png', mimetype='image/png')

@app.route('/icon-192.png')
def icon_192():
    return send_from_directory('.', 'icon-192.png', mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)