# predict.py
import pickle

import numpy as np
import pandas as pd
from typing import Dict, List, Union


class WaterPotabilityPredictor:
    """Water Potability Prediction Model"""
    
    def __init__(self, model_path='model.bin'):
        """
        Initialize the predictor by loading the trained model
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.pipeline = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f_in:
                model_data = pickle.load(f_in)
                
            # Handle both old and new model formats
            if isinstance(model_data, dict):
                self.pipeline = model_data['pipeline']
                self.feature_names = model_data['feature_names']
            else:
                # Legacy format (just the pipeline)
                self.pipeline = model_data
                self.feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 
                                     'Sulfate', 'Conductivity', 'Organic_carbon', 
                                     'Trihalomethanes', 'Turbidity']
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Expected features: {self.feature_names}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run train.py first to create the model."
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def validate_input(self, data: Dict[str, float]) -> pd.DataFrame:
        """
        Validate and prepare input data
        
        Args:
            data: Dictionary with feature names as keys and values as floats
            
        Returns:
            DataFrame with validated data
        """
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(data.keys())
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Required features: {self.feature_names}"
            )
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([data])[self.feature_names]
        
        # Validate data types and ranges
        for col in df.columns:
            if not isinstance(df[col].iloc[0], (int, float, np.number)):
                raise ValueError(f"Feature '{col}' must be numeric, got {type(df[col].iloc[0])}")
            
            # Check for reasonable ranges (optional validation)
            if df[col].iloc[0] < 0 and col not in ['ph']:  # pH can be any value, others should be positive
                raise ValueError(f"Feature '{col}' has negative value: {df[col].iloc[0]}")
        
        return df
    
    def predict(self, data: Dict[str, float]) -> Dict[str, Union[int, float, str]]:
        """
        Make prediction for a single sample
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        # Validate and prepare input
        df = self.validate_input(data)
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        probability = self.pipeline.predict_proba(df)[0, 1]
        
        # Determine status
        if probability >= 0.5:
            status = "Potable - Safe for Drinking"
            risk_level = "Low"
        elif probability >= 0.3:
            status = "Uncertain - Further Testing Recommended"
            risk_level = "Medium"
        else:
            status = "Not Potable - Do Not Drink"
            risk_level = "High"
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Potable' if prediction == 1 else 'Not Potable',
            'potability_probability': float(probability),
            'safety_status': status,
            'risk_level': risk_level,
            'confidence': float(max(probability, 1 - probability))
        }
    
    def predict_batch(self, data_list: List[Dict[str, float]]) -> List[Dict[str, Union[int, float, str]]]:
        """
        Make predictions for multiple samples
        
        Args:
            data_list: List of dictionaries with feature values
            
        Returns:
            List of prediction results
        """
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.predict(data)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'error': str(e)
                })
        
        return results


def main():
    """Example usage of the predictor"""
    # Initialize predictor
    predictor = WaterPotabilityPredictor('model.bin')
    
    # Example single prediction
    print("Single Sample Prediction")

    
    sample_data = {
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
    
    result = predictor.predict(sample_data)
    
    print(f"\nInput Data:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction Results:")
    print(f"  Prediction: {result['prediction_label']}")
    print(f"  Potability Probability: {result['potability_probability']:.4f}")
    print(f"  Status: {result['safety_status']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    # Example batch prediction
    print("\nBatch Prediction")
    
    batch_data = [
        {
            'ph': 8.0, 'Hardness': 200.0, 'Solids': 25000.0,
            'Chloramines': 8.0, 'Sulfate': 400.0, 'Conductivity': 450.0,
            'Organic_carbon': 16.0, 'Trihalomethanes': 75.0, 'Turbidity': 5.0
        },
        {
            'ph': 8.9, 'Hardness': 215.0, 'Solids': 15921.4,
            'Chloramines': 6.3, 'Sulfate': 312.9, 'Conductivity': 390.4,
            'Organic_carbon': 9.9, 'Trihalomethanes': 55.1, 'Turbidity': 4.6
        }
    ]

    
    batch_results = predictor.predict_batch(batch_data)
    
    for result in batch_results:
        if 'error' in result:
            print(f"\nSample {result['sample_id']}: ERROR - {result['error']}")
        else:
            print(f"\nSample {result['sample_id']}:")
            print(f"  Prediction: {result['prediction_label']}")
            print(f"  Probability: {result['potability_probability']:.4f}")
            print(f"  Status: {result['safety_status']}")


if __name__ == "__main__":
    main()