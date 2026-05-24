import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import datetime

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
CORS(app)

model = joblib.load('smart_brain.pkl')

@app.route('/')
def home():
    return "<h1>Smart Irrigation AI Server is Connected to the Cloud!</h1>"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        temp = float(request.args.get('temp'))
        hum = float(request.args.get('hum'))
        moist = float(request.args.get('moist'))
        
        input_data = np.array([[temp, hum, moist]])
        prediction = int(model.predict(input_data)[0])
        
        if prediction == 0:
            msg = "Normal: No Watering Needed"
        elif prediction == 1:
            msg = "Action: Watering Required"
        else:
            msg = "ALERT: Critical Soil Condition! (Level 2)"
            
        doc_ref = db.collection('irrigation_logs').document()
        doc_ref.set({
            'temperature': temp,
            'humidity': hum,
            'moisture': moist,
            'pump_action': prediction,
            'timestamp': datetime.datetime.now()
        })
            
        return jsonify({
            "status": "success",
            "environment": {
                "temperature": temp,
                "humidity": hum,
                "soil_moisture": moist
            },
            "pump_action": prediction, 
            "message": msg
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- NEW: Route to fetch history for the dashboard ---
@app.route('/history', methods=['GET'])
def history():
    try:
        # Fetch the 10 most recent logs from Firebase, sorted by time
        logs_ref = db.collection('irrigation_logs').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
        docs = logs_ref.stream()
        
        history_data = []
        for doc in docs:
            data = doc.to_dict()
            # Convert the complex timestamp into a readable format
            data['timestamp'] = data['timestamp'].strftime("%Y-%m-%d %I:%M %p")
            history_data.append(data)
            
        return jsonify({"status": "success", "data": history_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)