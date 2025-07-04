import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MillingParameterPredictor:
    """
    A machine learning system for predicting optimal milling parameters
    based on input material characteristics.
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.ensure_model_directory()
        
        # Model file paths
        self.model_files = {
            'classifier': os.path.join(model_dir, 'equipment_classifier.pkl'),
            'label_encoder': os.path.join(model_dir, 'equipment_label_encoder.pkl'),
            'scaler': os.path.join(model_dir, 'feature_scaler.pkl')
        }
        
        # Equipment types and their parameters
        self.equipment_types = ['Ball_Mill_A', 'Attritor_Mill_X']
        self.parameters = ['ball_filling_rate', 'rotation_speed', 'feed_rate', 'milling_time']
        
        # Add regression model file paths
        for eq in self.equipment_types:
            for param in self.parameters:
                filename = f"{eq}_{param}_model.pkl"
                self.model_files[f"{eq}_{param}"] = os.path.join(model_dir, filename)
        
        # Initialize models
        self.classifier = None
        self.label_encoder = None
        self.scaler = None
        self.regression_models = {}
        
    def ensure_model_directory(self):
        """Create models directory if it doesn't exist."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Created directory: {self.model_dir}")
    
    def generate_synthetic_data(self, n_samples=500):
        """Generate synthetic training data with improved realism."""
        np.random.seed(42)  # For reproducibility
        
        # Generate features with different distributions - EXPANDED RANGES
        particle_size = np.random.gamma(2, 15) + 5   # Expanded particle size distribution
        weight = np.random.exponential(30) + 1       # Expanded batch weight distribution
        temperature = np.random.normal(25, 15)       # Expanded temperature range
        
        # Combine features with expanded ranges
        X = np.column_stack([
            np.random.gamma(2, 20, n_samples) + 5,   # particle size (5-200 μm)
            np.random.exponential(30, n_samples) + 1, # weight (1-500 kg)
            np.random.normal(25, 15, n_samples)      # temperature (-10-60°C)
        ])
        
        # Clip values to expanded but realistic ranges
        X[:, 0] = np.clip(X[:, 0], 5, 200)   # particle size - expanded
        X[:, 1] = np.clip(X[:, 1], 1, 500)   # weight - expanded
        X[:, 2] = np.clip(X[:, 2], -10, 60)  # temperature - expanded
        
        # More realistic equipment selection logic
        # Ball Mill A: Better for smaller particles and lighter batches
        # Attritor Mill X: Better for larger particles and heavier batches
        ball_mill_prob = 1 / (1 + np.exp(0.1 * (X[:, 0] - 40) + 0.05 * (X[:, 1] - 30)))
        y = np.where(np.random.random(n_samples) < ball_mill_prob, 'Ball_Mill_A', 'Attritor_Mill_X')
        
        return X, y
    
    def generate_parameter_targets(self, X, equipment_type):
        """Generate realistic parameter targets for given equipment type."""
        n_samples = X.shape[0]
        
        if equipment_type == 'Ball_Mill_A':
            # Ball Mill A parameters
            ball_filling_rate = 30 + 0.2 * X[:, 0] + 0.1 * X[:, 1] + np.random.normal(0, 3, n_samples)
            rotation_speed = 800 + 2 * X[:, 0] - 1 * X[:, 1] + np.random.normal(0, 20, n_samples)
            feed_rate = 5 + 0.1 * X[:, 1] + 0.02 * X[:, 0] + np.random.normal(0, 1, n_samples)
            milling_time = 45 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + np.random.normal(0, 5, n_samples)
        else:  # Attritor_Mill_X
            # Attritor Mill X parameters
            ball_filling_rate = 25 + 0.15 * X[:, 0] + 0.15 * X[:, 1] + np.random.normal(0, 3, n_samples)
            rotation_speed = 1200 - 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 30, n_samples)
            feed_rate = 8 + 0.15 * X[:, 1] + 0.01 * X[:, 0] + np.random.normal(0, 1.5, n_samples)
            milling_time = 35 - 0.2 * X[:, 0] + 0.1 * X[:, 1] + np.random.normal(0, 4, n_samples)
        
        # Clip parameters to realistic ranges
        ball_filling_rate = np.clip(ball_filling_rate, 10, 60)
        rotation_speed = np.clip(rotation_speed, 200, 2000)
        feed_rate = np.clip(feed_rate, 1, 20)
        milling_time = np.clip(milling_time, 10, 120)
        
        return {
            'ball_filling_rate': ball_filling_rate,
            'rotation_speed': rotation_speed,
            'feed_rate': feed_rate,
            'milling_time': milling_time
        }
    
    def train_models(self, n_samples=500):
        """Train all models and save them."""
        print("Generating synthetic training data...")
        X, y = self.generate_synthetic_data(n_samples)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train classification model
        print("Training equipment classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n=== Equipment Classifier Performance ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Cross-validation score: {cross_val_score(self.classifier, X_scaled, y_encoded, cv=5).mean():.3f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Train regression models for each equipment type
        for equipment in self.equipment_types:
            print(f"\nTraining regression models for {equipment}...")
            
            # Filter data for this equipment
            eq_mask = (y == equipment)
            X_eq = X[eq_mask]
            X_eq_scaled = X_scaled[eq_mask]
            
            # Generate parameter targets
            param_targets = self.generate_parameter_targets(X_eq, equipment)
            
            # Train regression model for each parameter
            for param_name, param_values in param_targets.items():
                print(f"  Training {param_name} model...")
                
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                    X_eq_scaled, param_values, test_size=0.2, random_state=42
                )
                
                # Use pipeline for regression
                reg_pipeline = Pipeline([
                    ('regressor', LinearRegression())
                ])
                
                reg_pipeline.fit(X_train_r, y_train_r)
                y_pred_r = reg_pipeline.predict(X_test_r)
                
                mse = mean_squared_error(y_test_r, y_pred_r)
                r2 = r2_score(y_test_r, y_pred_r)
                
                print(f"    {param_name}: MSE = {mse:.3f}, R² = {r2:.3f}")
                
                # Store model
                self.regression_models[f"{equipment}_{param_name}"] = reg_pipeline
        
        # Save all models
        self.save_models()
        print("\nAll models trained and saved successfully!")
    
    def save_models(self):
        """Save all trained models to disk."""
        joblib.dump(self.classifier, self.model_files['classifier'])
        joblib.dump(self.label_encoder, self.model_files['label_encoder'])
        joblib.dump(self.scaler, self.model_files['scaler'])
        
        for key, model in self.regression_models.items():
            joblib.dump(model, self.model_files[key])
    
    def load_models(self):
        """Load all models from disk."""
        try:
            self.classifier = joblib.load(self.model_files['classifier'])
            self.label_encoder = joblib.load(self.model_files['label_encoder'])
            self.scaler = joblib.load(self.model_files['scaler'])
            
            for eq in self.equipment_types:
                for param in self.parameters:
                    key = f"{eq}_{param}"
                    self.regression_models[key] = joblib.load(self.model_files[key])
            
            print("All models loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False
    
    def models_exist(self):
        """Check if all model files exist."""
        return all(os.path.exists(file) for file in self.model_files.values())
    
    def predict_parameters(self, particle_size, batch_weight, temperature):
        """Predict optimal milling parameters for given inputs."""
        # Validate inputs with expanded ranges
        if not (5 <= particle_size <= 200):
            raise ValueError("Particle size must be between 5 and 200 micrometers")
        if not (1 <= batch_weight <= 500):
            raise ValueError("Batch weight must be between 1 and 500 kg")
        if not (-10 <= temperature <= 60):
            raise ValueError("Temperature must be between -10 and 60°C")
        
        # Prepare features
        features = np.array([[particle_size, batch_weight, temperature]])
        features_scaled = self.scaler.transform(features)
        
        # Predict equipment
        pred_eq_idx = self.classifier.predict(features_scaled)[0]
        selected_equipment = self.label_encoder.inverse_transform([pred_eq_idx])[0]
        
        # Get prediction probabilities
        probs = self.classifier.predict_proba(features_scaled)[0]
        prob_map = {eq: prob for eq, prob in zip(self.label_encoder.classes_, probs)}
        
        # Predict parameters using regression models
        predictions = {}
        for param in self.parameters:
            model_key = f"{selected_equipment}_{param}"
            model = self.regression_models[model_key]
            pred_value = model.predict(features_scaled)[0]
            predictions[param] = pred_value
        
        return {
            'equipment': selected_equipment,
            'equipment_probabilities': prob_map,
            'parameters': predictions,
            'input_features': {
                'particle_size': particle_size,
                'batch_weight': batch_weight,
                'temperature': temperature
            }
        }
    
    def get_fmea_info(self):
        """Return FMEA (Failure Mode and Effects Analysis) information."""
        return {
            'Ball Filling Rate': {
                'risk': 'Inefficient grinding, high energy consumption, or equipment damage if set improperly',
                'control': 'Monitor ball-to-powder ratio, ensure uniform ball distribution, check for wear',
                'acceptable_range': '10-60%'
            },
            'Rotation Speed': {
                'risk': 'Overheating, excessive wear, or poor milling efficiency if speed is incorrect',
                'control': 'Monitor motor load, material temperature, and vibration levels',
                'acceptable_range': '200-2000 RPM'
            },
            'Feed Rate': {
                'risk': 'Material segregation, inconsistent particle size, or equipment overload',
                'control': 'Ensure consistent feed flow, monitor particle size distribution',
                'acceptable_range': '1-20 kg/min'
            },
            'Milling Time': {
                'risk': 'Incomplete milling, excessive energy use, or contamination from prolonged operation',
                'control': 'Regular particle size sampling, monitor energy consumption patterns',
                'acceptable_range': '10-120 minutes'
            }
        }
    
    def print_prediction_report(self, prediction_result):
        """Print a formatted prediction report."""
        result = prediction_result
        
        print("\n" + "="*60)
        print("MILLING PARAMETER PREDICTION REPORT")
        print("="*60)
        
        print(f"\nInput Parameters:")
        print(f"  • Particle Size: {result['input_features']['particle_size']:.1f} μm")
        print(f"  • Batch Weight: {result['input_features']['batch_weight']:.1f} kg")
        print(f"  • Temperature: {result['input_features']['temperature']:.1f} °C")
        
        print(f"\nRecommended Equipment: {result['equipment']}")
        print(f"Confidence: {result['equipment_probabilities'][result['equipment']]*100:.1f}%")
        
        print(f"\nOptimal Parameters:")
        print(f"  • Ball Filling Rate: {result['parameters']['ball_filling_rate']:.1f}%")
        print(f"  • Rotation Speed: {result['parameters']['rotation_speed']:.0f} RPM")
        print(f"  • Feed Rate: {result['parameters']['feed_rate']:.1f} kg/min")
        print(f"  • Milling Time: {result['parameters']['milling_time']:.1f} minutes")
        
        print(f"\nEquipment Selection Probabilities:")
        for eq, prob in result['equipment_probabilities'].items():
            print(f"  • {eq}: {prob*100:.1f}%")
        
        print(f"\nFMEA & Control Recommendations:")
        fmea = self.get_fmea_info()
        for param_name, info in fmea.items():
            print(f"\n{param_name}:")
            print(f"  Risk: {info['risk']}")
            print(f"  Control: {info['control']}")
            print(f"  Range: {info['acceptable_range']}")


def main():
    """Main execution function."""
    print("Milling Parameter Prediction System")
    print("="*50)
    
    # Initialize predictor
    predictor = MillingParameterPredictor()
    
    # Check if models exist, if not train them
    if not predictor.models_exist():
        print("Models not found. Training new models...")
        predictor.train_models(n_samples=1000)
    else:
        print("Loading existing models...")
        predictor.load_models()
    
    # Get user input
    print("\n" + "="*50)
    print("PARAMETER INPUT")
    print("="*50)
    
    try:
        particle_size = float(input("Enter target particle size (5-200 μm): "))
        batch_weight = float(input("Enter batch weight (1-500 kg): "))
        temperature = float(input("Enter process temperature (-10-60°C): "))
        
        # Make prediction
        result = predictor.predict_parameters(particle_size, batch_weight, temperature)
        
        # Print results
        predictor.print_prediction_report(result)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please enter valid numeric values within the specified ranges.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
