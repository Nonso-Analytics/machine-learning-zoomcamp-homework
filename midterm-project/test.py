import requests
import json

# Define the prediction endpoint URL
# Use http://localhost:8000/predict if you're running the Flask app locally
#url = 'http://localhost:8000/predict'
url = 'https://broken-wind-2125.fly.dev/predict'

water_sample = {
    'ph': 7.5,
    'Hardness': 180.0,
    'Solids': 20000.0,
    'Chloramines': 7.5,
    'Sulfate': 350.0,
    'Conductivity': 400.0,
    'Organic_carbon': 15.0,
    'Trihalomethanes': 70.0,
    'Turbidity': 4.0
}

# --- Make the API Request ---
try:
    print(f"Sending request to: {url}")
    print(f"Input data: {json.dumps(water_sample, indent=4)}")

    response = requests.post(url, json=water_sample, timeout=10)
    response.raise_for_status()
    
    predictions = response.json()

    print("\n--- Prediction Results ---")
    
    # The output structure is expected to be similar to the return of predictor.predict:
    # {'prediction': int, 'prediction_label': str, 'potability_probability': float, 'safety_status': str, 'risk_level': str, 'confidence': float}
    
    # Check for the key that indicates the final prediction (1 for Potable, 0 for Not Potable)
    if predictions.get('prediction') == 1:
        print(f"The water sample is Potable (Safe for Drinking).")
    elif predictions.get('prediction') == 0:
        print(f"The water sample is Not Potable (Do Not Drink).")
    else:
        # Handle cases where the response structure is unexpected
        print("Prediction key not found in response or unexpected value.")
    
    # Print detailed results if available
    if 'safety_status' in predictions:
        print(f"Status: {predictions['safety_status']}")
    if 'potability_probability' in predictions:
        print(f"Potability Probability: {predictions['potability_probability']:.4f}")
    if 'risk_level' in predictions:
        print(f"Risk Level: {predictions['risk_level']}")
    
    # Print the full JSON response for verification
    print("\nFull API Response:")
    print(json.dumps(predictions, indent=4))

except requests.exceptions.ConnectionError:
    print("\n**Connection Error:** Could not connect to the API. Make sure your Flask app is running at `http://localhost:8000`.")
except requests.exceptions.Timeout:
    print("\n**Timeout Error:** The request took too long to complete. The server may be overloaded or non-responsive.")
except requests.exceptions.RequestException as e:
    print(f"\n**An error occurred during the request:** {e}")