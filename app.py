import os
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)

MODEL_FILE = 'railway_model_1cr.pkl'
try:
    model = joblib.load(MODEL_FILE)
except:
    model = None

HIGH_DEMAND_CODES = {'NDLS', 'HWH', 'CSMT', 'PNBE', 'PRYJ', 'ERS'}

STATION_STATES = {
    'PNBE': 'Bihar', 'MFP': 'Bihar', 'DBG': 'Bihar', 'GAYA': 'Bihar',
    'ERS': 'Kerala', 'TVC': 'Kerala', 'KTYM': 'Kerala', 'TCR': 'Kerala',
    'HWH': 'West_Bengal', 'SDAH': 'West_Bengal', 'KGP': 'West_Bengal',
    'CSMT': 'Maharashtra', 'PUNE': 'Maharashtra', 'NGP': 'Maharashtra'
}

STATE_FESTIVALS = {
    'Bihar': [(10, 11)],
    'Kerala': [(8, 9)],
    'West_Bengal': [(9, 10)],
    'Maharashtra': [(8, 9)]
}

def evaluate_route_tier(src_code, dest_code):
    if src_code in HIGH_DEMAND_CODES or dest_code in HIGH_DEMAND_CODES:
        return 1
    return 2

def calculate_days_left(date_str):
    journey = datetime.strptime(date_str, '%Y-%m-%d')
    return (journey - datetime.today()).days

def check_local_festival(src_code, dest_code, journey_date):
    dt = datetime.strptime(journey_date, '%Y-%m-%d')
    month = dt.month
    
    states_involved = set()
    if src_code in STATION_STATES:
        states_involved.add(STATION_STATES[src_code])
    if dest_code in STATION_STATES:
        states_involved.add(STATION_STATES[dest_code])
        
    for state in states_involved:
        if state in STATE_FESTIVALS:
            for start_mo, end_mo in STATE_FESTIVALS[state]:
                if start_mo <= month <= end_mo:
                    return 1
    return 0

@app.route('/')
def home():
    return render_template('index.html')
def predict():
    try:
        data = request.get_json()
        
        src_code = data.get('source_station', '')
        dest_code = data.get('destination_station', '')
        journey_date = data.get('journey_date', datetime.today().strftime('%Y-%m-%d'))
        waitlist_num = int(data.get('waitlist_number', 10))
        travel_class = int(data.get('coach_type', 1))
        quota = int(data.get('quota', 0))
        train_type = 0 

        days_left = calculate_days_left(journey_date)

        if days_left <= 2 and travel_class in [2, 3] and waitlist_num >= 3:
            prob_rounded = 2.5
            status = "High Risk of Waitlist / Cancellation"
            
        elif days_left <= 2 and waitlist_num >= 10:
            prob_rounded = 4.5
            status = "High Risk of Waitlist / Cancellation"
            
        else:
            route_tier = evaluate_route_tier(src_code, dest_code)
            is_festival = check_local_festival(src_code, dest_code, journey_date)

            features = pd.DataFrame([[route_tier, days_left, waitlist_num, train_type, travel_class, quota, is_festival]],
                                    columns=['route_tier', 'days_left', 'waitlist_num', 'train_type', 'class', 'quota', 'is_festival'])
            
            if model:
                base_prob = model.predict_proba(features)[0][1] * 100
            else:
                base_prob = 75.0
            
            if days_left > 30:
                wl_multiplier = 0.8
            elif days_left > 10:
                wl_multiplier = 1.5
            else:
                wl_multiplier = 3.5
                
            waitlist_decay = waitlist_num * wl_multiplier
            days_bonus = (days_left / 60.0) * 12.0 

            class_penalty = {0: 8.0, 1: 0.0, 2: -25.0, 3: -45.0}.get(travel_class, 0.0)
            regional_penalty = -25.0 if is_festival == 1 else 0.0

            prediction_prob = base_prob - waitlist_decay + days_bonus + class_penalty + regional_penalty
            
            prediction_prob = np.clip(prediction_prob, 1.5, 98.5)
            prob_rounded = round(float(prediction_prob), 2)
            
            status = "High Chance of Confirmation" if prob_rounded > 50 else "High Risk of Waitlist / Cancellation"

        return jsonify({'success': True, 'probability': prob_rounded, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
