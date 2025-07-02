import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class Dismizer:
    """
    A system that learns from dispersion process data to recommend optimal
    equipment and RPM. It aims to minimize energy costs and provide analysis
    based on a built-in FMEA and Control Plan.
    
    Features:
    - Equipment-specific modeling
    - Model validation and performance metrics
    - Data preprocessing and outlier detection
    - Improved error handling
    """
    def __init__(self, data_file='dispersion_data.csv', model_path='models/'):
        self.data_file = data_file
        self.model_path = model_path
        self.version = "2.0"
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.data = self.load_data()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Equipment-specific models
        self.equipment_models = {}
        self.equipment_list = []
        
        # Global models
        self.equipment_selector_model = None
        self.global_viscosity_model = None
        
        # Model performance metrics
        self.model_metrics = {}
        
        self.load_models()
        self.fmea_df, self.control_plan_df = self.generate_fmea_control_plan()

    def load_data(self):
        """Loads data from a CSV file or creates a new DataFrame if the file doesn't exist."""
        if os.path.exists(self.data_file):
            print(f"Loading existing data from '{self.data_file}'.")
            data = pd.read_csv(self.data_file)
            print(f"Loaded {len(data)} records.")
            return data
        else:
            print("Data file not found. Creating a new one.")
            return pd.DataFrame(columns=[
                'Initial_Viscosity', 'Weight', 'Equipment', 'RPM',
                'Agitation_Time', 'Final_Viscosity', 'Temperature', 'Batch_ID'
            ])

    def add_data(self, initial_viscosity, weight, equipment, rpm, agitation_time, 
                 final_viscosity, temperature=25, batch_id=None):
        """Adds new process data with validation and saves it to the CSV file."""
        
        # Data validation
        if not self._validate_input_data(initial_viscosity, weight, equipment, rpm, 
                                       agitation_time, final_viscosity, temperature):
            return False
        
        if batch_id is None:
            batch_id = f"BATCH_{len(self.data) + 1:04d}"
        
        new_data = pd.DataFrame([{
            'Initial_Viscosity': initial_viscosity,
            'Weight': weight,
            'Equipment': equipment,
            'RPM': rpm,
            'Agitation_Time': agitation_time,
            'Final_Viscosity': final_viscosity,
            'Temperature': temperature,
            'Batch_ID': batch_id
        }])
        
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.data.to_csv(self.data_file, index=False)
        print(f"New data added successfully. Total records: {len(self.data)}")
        return True

    def _validate_input_data(self, initial_viscosity, weight, equipment, rpm, 
                           agitation_time, final_viscosity, temperature):
        """Validates input data ranges and types."""
        try:
            # Range validations
            if not (10 <= initial_viscosity <= 100000):
                print("Error: Initial viscosity should be between 10-100,000 cP")
                return False
            if not (50 <= weight <= 5000):
                print("Error: Weight should be between 50-5,000 kg")
                return False
            if not (100 <= rpm <= 2000):
                print("Error: RPM should be between 100-2,000")
                return False
            if not (5 <= agitation_time <= 300):
                print("Error: Agitation time should be between 5-300 minutes")
                return False
            if not (10 <= final_viscosity <= 100000):
                print("Error: Final viscosity should be between 10-100,000 cP")
                return False
            if not (10 <= temperature <= 80):
                print("Error: Temperature should be between 10-80°C")
                return False
            
            return True
        except (TypeError, ValueError):
            print("Error: Invalid data types provided")
            return False

    def _calculate_energy_cost(self, rpm, time, weight, equipment):
        """Estimates the energy cost with equipment-specific factors."""
        # Equipment efficiency factors
        equipment_factors = {
            'Mixer-A': 1.0,
            'Mixer-B': 1.2,
            'Mixer-C': 0.8
        }
        
        base_constant = 0.0001
        equipment_factor = equipment_factors.get(equipment, 1.0)
        
        cost = base_constant * (rpm ** 2) * time * weight * equipment_factor
        return cost

    def preprocess_data(self):
        """Preprocesses data including outlier detection and feature engineering."""
        if len(self.data) < 5:
            print("Insufficient data for preprocessing.")
            return None
        
        # Create a copy for preprocessing
        processed_data = self.data.copy()
        
        # Feature engineering
        processed_data['Viscosity_Ratio'] = processed_data['Final_Viscosity'] / processed_data['Initial_Viscosity']
        processed_data['Intensity_Factor'] = processed_data['RPM'] * processed_data['Agitation_Time']
        processed_data['Weight_per_RPM'] = processed_data['Weight'] / processed_data['RPM']
        
        # Outlier detection using IQR method
        numeric_columns = ['Initial_Viscosity', 'Weight', 'RPM', 'Agitation_Time', 'Final_Viscosity']
        outlier_indices = set()
        
        for col in numeric_columns:
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = processed_data[(processed_data[col] < lower_bound) | 
                                    (processed_data[col] > upper_bound)].index
            outlier_indices.update(outliers)
        
        if outlier_indices:
            print(f"Detected {len(outlier_indices)} potential outliers.")
            # Remove outliers (optional - can be made configurable)
            processed_data = processed_data.drop(outlier_indices).reset_index(drop=True)
        
        return processed_data

    def train_models(self):
        """Trains equipment-specific and global models with validation."""
        processed_data = self.preprocess_data()
        if processed_data is None or len(processed_data) < 10:
            print("Insufficient data for training. At least 10 clean data points are required.")
            return False

        print("Starting model training...")
        
        # Encode equipment names
        equipment_encoded = self.label_encoder.fit_transform(processed_data['Equipment'])
        self.equipment_list = list(processed_data['Equipment'].unique())
        
        try:
            # Train equipment selector (global model)
            X_equip = processed_data[['Initial_Viscosity', 'Weight', 'Temperature']]
            y_equip = equipment_encoded
            
            if len(np.unique(y_equip)) > 1:  # Only if multiple equipment types
                self.equipment_selector_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
                self.equipment_selector_model.fit(X_equip, y_equip)
                
                # Evaluate equipment selector
                scores = cross_val_score(self.equipment_selector_model, X_equip, y_equip, cv=3)
                self.model_metrics['equipment_selector_accuracy'] = scores.mean()
            
            # Train equipment-specific models
            for equipment in self.equipment_list:
                equip_data = processed_data[processed_data['Equipment'] == equipment]
                
                if len(equip_data) < 5:
                    print(f"Insufficient data for {equipment} (only {len(equip_data)} records)")
                    continue
                
                print(f"Training models for {equipment}...")
                
                # Features for equipment-specific models
                X_base = equip_data[['Initial_Viscosity', 'Weight', 'Temperature']]
                
                # RPM model
                y_rpm = equip_data['RPM']
                rpm_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rpm_model.fit(X_base, y_rpm)
                
                # Time model
                X_time = equip_data[['Initial_Viscosity', 'Weight', 'RPM', 'Temperature']]
                y_time = equip_data['Agitation_Time']
                time_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                time_model.fit(X_time, y_time)
                
                # Viscosity model
                X_visc = equip_data[['Initial_Viscosity', 'Weight', 'RPM', 'Agitation_Time', 'Temperature']]
                y_visc = equip_data['Final_Viscosity']
                visc_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                visc_model.fit(X_visc, y_visc)
                
                # Store models
                self.equipment_models[equipment] = {
                    'rpm_model': rpm_model,
                    'time_model': time_model,
                    'viscosity_model': visc_model
                }
                
                # Evaluate models
                if len(equip_data) >= 5:
                    rpm_scores = cross_val_score(rpm_model, X_base, y_rpm, cv=3, scoring='neg_mean_squared_error')
                    time_scores = cross_val_score(time_model, X_time, y_time, cv=3, scoring='neg_mean_squared_error')
                    visc_scores = cross_val_score(visc_model, X_visc, y_visc, cv=3, scoring='neg_mean_squared_error')
                    
                    self.model_metrics[f'{equipment}_rpm_rmse'] = np.sqrt(-rpm_scores.mean())
                    self.model_metrics[f'{equipment}_time_rmse'] = np.sqrt(-time_scores.mean())
                    self.model_metrics[f'{equipment}_viscosity_rmse'] = np.sqrt(-visc_scores.mean())
            
            # Save all models
            self._save_models()
            
            print("Model training completed successfully!")
            self._print_model_performance()
            return True
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return False

    def _save_models(self):
        """Saves all trained models to files."""
        try:
            # Save equipment-specific models
            for equipment, models in self.equipment_models.items():
                equip_path = os.path.join(self.model_path, equipment.replace('-', '_').lower())
                os.makedirs(equip_path, exist_ok=True)
                
                for model_name, model in models.items():
                    joblib.dump(model, os.path.join(equip_path, f'{model_name}.joblib'))
            
            # Save global models
            if self.equipment_selector_model:
                joblib.dump(self.equipment_selector_model, 
                           os.path.join(self.model_path, 'equipment_selector.joblib'))
            
            joblib.dump(self.label_encoder, os.path.join(self.model_path, 'label_encoder.joblib'))
            joblib.dump(self.model_metrics, os.path.join(self.model_path, 'model_metrics.joblib'))
            joblib.dump(self.equipment_list, os.path.join(self.model_path, 'equipment_list.joblib'))
            
            print("All models saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")

    def load_models(self):
        """Loads saved models from files."""
        try:
            # Load basic components
            if os.path.exists(os.path.join(self.model_path, 'label_encoder.joblib')):
                self.label_encoder = joblib.load(os.path.join(self.model_path, 'label_encoder.joblib'))
                self.equipment_list = joblib.load(os.path.join(self.model_path, 'equipment_list.joblib'))
                self.model_metrics = joblib.load(os.path.join(self.model_path, 'model_metrics.joblib'))
                
                # Load equipment selector
                equip_selector_path = os.path.join(self.model_path, 'equipment_selector.joblib')
                if os.path.exists(equip_selector_path):
                    self.equipment_selector_model = joblib.load(equip_selector_path)
                
                # Load equipment-specific models
                for equipment in self.equipment_list:
                    equip_path = os.path.join(self.model_path, equipment.replace('-', '_').lower())
                    if os.path.exists(equip_path):
                        models = {}
                        for model_name in ['rpm_model', 'time_model', 'viscosity_model']:
                            model_file = os.path.join(equip_path, f'{model_name}.joblib')
                            if os.path.exists(model_file):
                                models[model_name] = joblib.load(model_file)
                        
                        if models:
                            self.equipment_models[equipment] = models
                
                print(f"Successfully loaded models for {len(self.equipment_models)} equipment types.")
                return True
            else:
                print("No saved models found. Please train models first.")
                return False
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def _print_model_performance(self):
        """Prints model performance metrics."""
        if not self.model_metrics:
            return
        
        print("\n--- Model Performance Metrics ---")
        for metric_name, value in self.model_metrics.items():
            if 'accuracy' in metric_name:
                print(f"{metric_name}: {value:.3f}")
            else:
                print(f"{metric_name}: {value:.2f}")
        print("--------------------------------")

    def recommend_optimized_process(self, initial_viscosity, weight, temperature=25, target_equipment=None):
        """
        Recommends optimal process parameters with equipment-specific modeling.
        """
        if not self.equipment_models:
            print("Models are not ready. Please train the models first.")
            return None
        
        # Validate inputs
        if not self._validate_input_data(initial_viscosity, weight, 'Mixer-A', 800, 60, 15000, temperature):
            return None
        
        print(f"\n[Optimization Search] Initial Viscosity: {initial_viscosity} cP, Weight: {weight} kg, Temperature: {temperature}°C")
        
        # Determine target equipment
        if target_equipment and target_equipment in self.equipment_models:
            equipment_candidates = [target_equipment]
        else:
            if self.equipment_selector_model and len(self.equipment_list) > 1:
                # Use model to suggest best equipment
                equip_pred = self.equipment_selector_model.predict([[initial_viscosity, weight, temperature]])
                suggested_equipment = self.label_encoder.inverse_transform(equip_pred)[0]
                equipment_candidates = [suggested_equipment]
            else:
                # Try all available equipment
                equipment_candidates = list(self.equipment_models.keys())
        
        best_results = []
        
        for equipment in equipment_candidates:
            if equipment not in self.equipment_models:
                continue
            
            models = self.equipment_models[equipment]
            
            try:
                # Get base prediction
                base_features = [[initial_viscosity, weight, temperature]]
                base_rpm = models['rpm_model'].predict(base_features)[0]
                
                # Test RPM variations
                rpm_candidates = np.linspace(max(100, base_rpm * 0.8), 
                                           min(2000, base_rpm * 1.2), 5)
                
                for rpm in rpm_candidates:
                    rpm = round(rpm)
                    
                    # Predict time
                    time_features = [[initial_viscosity, weight, rpm, temperature]]
                    predicted_time = models['time_model'].predict(time_features)[0]
                    
                    # Predict final viscosity
                    visc_features = [[initial_viscosity, weight, rpm, predicted_time, temperature]]
                    predicted_viscosity = models['viscosity_model'].predict(visc_features)[0]
                    
                    # Calculate energy cost
                    energy_cost = self._calculate_energy_cost(rpm, predicted_time, weight, equipment)
                    
                    # Calculate efficiency score (lower is better)
                    efficiency_score = energy_cost + abs(predicted_viscosity - initial_viscosity) * 0.001
                    
                    best_results.append({
                        'equipment': equipment,
                        'rpm': rpm,
                        'estimated_time': round(predicted_time, 2),
                        'estimated_viscosity': round(predicted_viscosity),
                        'estimated_energy_cost': round(energy_cost, 2),
                        'efficiency_score': round(efficiency_score, 4)
                    })
                    
            except Exception as e:
                print(f"Error processing {equipment}: {str(e)}")
                continue
        
        if not best_results:
            print("Could not generate recommendations.")
            return None
        
        # Select best option based on efficiency score
        best_process = min(best_results, key=lambda x: x['efficiency_score'])
        
        return best_process

    def get_equipment_statistics(self):
        """Returns statistics about equipment usage and performance."""
        if len(self.data) == 0:
            return None
        
        stats = {}
        for equipment in self.data['Equipment'].unique():
            equip_data = self.data[self.data['Equipment'] == equipment]
            stats[equipment] = {
                'count': len(equip_data),
                'avg_rpm': equip_data['RPM'].mean(),
                'avg_time': equip_data['Agitation_Time'].mean(),
                'avg_viscosity_change': (equip_data['Final_Viscosity'] - equip_data['Initial_Viscosity']).mean()
            }
        
        return stats

    def generate_fmea_control_plan(self):
        """Generates an enhanced FMEA and Control Plan for the dispersion process."""
        print("\nGenerating enhanced FMEA and Control Plan for the dispersion process.")

        fmea_data = {
            'Process Step': [
                'Raw Material Weighing', 'Raw Material Charging', 'Temperature Control',
                'Dispersion Agitation', 'Dispersion Agitation', 'Equipment Selection',
                'Quality Inspection', 'Data Recording'
            ],
            'Potential Failure Mode': [
                'Weighing Error', 'Incorrect Charging Order', 'Temperature Deviation',
                'Insufficient Agitation (Low RPM/Short Time)', 'Over-dispersion (High RPM/Long Time)',
                'Wrong Equipment Selection', 'Viscosity Measurement Error', 'Data Loss/Corruption'
            ],
            'Potential Effects of Failure': [
                'Defective Product Properties', 'Poor Dispersion, Gelling', 'Inconsistent Results',
                'Agglomerates, Poor Dispersion', 'Reduced Product Stability', 'Inefficient Process',
                'Incorrect Quality Judgment', 'Loss of Process Knowledge'
            ],
            'S (Severity)': [8, 7, 6, 9, 7, 5, 8, 4],
            'Potential Cause(s)': [
                'Scale Malfunction/Error', 'Operator Mistake', 'Heating/Cooling System Failure',
                'Incorrect RPM/Time Setting', 'Incorrect RPM/Time Setting', 'Poor Equipment Selection',
                'Viscometer Error/Deviation', 'System/Database Error'
            ],
            'O (Occurrence)': [3, 2, 4, 5, 4, 3, 3, 2],
            'Current Process Controls': [
                'Regular Scale Calibration', 'Work Instruction (SOP)', 'Temperature Monitoring',
                'SOP & Operator Training', 'SOP & Operator Training', 'Equipment Selection Guide',
                'Viscometer Calibration', 'Automated Data Backup'
            ],
            'D (Detection)': [4, 3, 3, 6, 6, 5, 4, 2],
            'Recommended Actions': [
                'Implement Automated Weighing', 'Systematize Charging Sequence', 'Enhanced Temperature Control',
                'Use AI-Based RPM/Time Optimization', 'Use AI-Based RPM/Time Optimization',
                'Implement Equipment Selection AI', 'Automate Measurement & Validation', 'Redundant Data Storage'
            ]
        }
        
        fmea_df = pd.DataFrame(fmea_data)
        fmea_df['RPN'] = fmea_df['S (Severity)'] * fmea_df['O (Occurrence)'] * fmea_df['D (Detection)']
        
        control_plan_data = {
            'Process Step': [
                'Raw Material Weighing', 'Temperature Control', 'Equipment Selection',
                'Dispersion Agitation', 'Dispersion Agitation', 'Quality Inspection', 'Process Monitoring'
            ],
            'Control Item': [
                'Material Weight', 'Process Temperature', 'Equipment Type',
                'Agitation RPM', 'Agitation Time', 'Final Viscosity', 'Model Performance'
            ],
            'Specification': [
                'Target Weight ±0.5%', 'Target Temperature ±2°C', 'AI Recommendation',
                'AI Recommendation ±5%', 'AI Recommendation ±5%', 'Target Viscosity ±10%', 'RMSE < 10%'
            ],
            'Measurement Method': [
                'Electronic Scale', 'Temperature Sensor', 'Equipment Database',
                'Equipment Control Panel', 'Timer/PLC', 'Viscometer', 'Model Validation'
            ],
            'Frequency': [
                'Every Batch', 'Continuous', 'Every Batch',
                'Real-time Monitoring', 'Real-time Monitoring', 'After Batch Completion', 'Weekly'
            ],
            'Control Method': [
                'Operator Check & Record', 'Automatic Control', 'AI System Selection',
                'AI System Setting & Verification', 'AI System Setting & Verification',
                'QC Personnel Measurement', 'Automated Model Check'
            ],
            'Reaction Plan': [
                'Re-weigh', 'Adjust Temperature', 'Override with Manual Selection',
                'Re-check/Correct Settings', 'Re-check/Correct Settings',
                'Review for Rework or Scrap', 'Retrain Model'
            ]
        }
        
        control_plan_df = pd.DataFrame(control_plan_data)
        
        return fmea_df, control_plan_df

    def analyze_recommendation_with_fmea(self, recommendation):
        """Enhanced analysis of recommendations with FMEA context."""
        if recommendation is None:
            return

        print("\n--- Enhanced FMEA & Control Plan Analysis ---")
        
        rec_equipment = recommendation['equipment']
        rec_rpm = recommendation['rpm']
        rec_time = recommendation['estimated_time']
        rec_viscosity = recommendation['estimated_viscosity']

        print(f"\n[Recommended Process Parameters]")
        print(f"  Equipment: {rec_equipment}")
        print(f"  RPM: {rec_rpm}")
        print(f"  Time: {rec_time} min")
        print(f"  Expected Final Viscosity: {rec_viscosity} cP")
        print(f"  Efficiency Score: {recommendation.get('efficiency_score', 'N/A')}")
        
        # Equipment-specific analysis
        if rec_equipment in self.equipment_models:
            print(f"\n[Equipment-Specific Model Confidence]")
            model_key = f"{rec_equipment}_viscosity_rmse"
            if model_key in self.model_metrics:
                rmse = self.model_metrics[model_key]
                confidence = max(0, min(100, 100 - (rmse / rec_viscosity * 100)))
                print(f"  Model RMSE: {rmse:.2f} cP")
                print(f"  Prediction Confidence: {confidence:.1f}%")

        print("\n[Control Plan Compliance Check]")
        control_items = self.control_plan_df[
            self.control_plan_df['Control Item'].isin(['Equipment Type', 'Agitation RPM', 'Agitation Time'])
        ]
        
        for _, row in control_items.iterrows():
            item = row['Control Item']
            if 'Equipment' in item:
                value = rec_equipment
                unit = ''
            elif 'RPM' in item:
                value = rec_rpm
                unit = ' RPM'
            else:
                value = rec_time
                unit = ' min'
            
            print(f"  ✓ {item}: {value}{unit}")
            print(f"    Specification: {row['Specification']}")
        
        print("\n[Risk Mitigation Analysis]")
        high_risk_items = self.fmea_df[self.fmea_df['RPN'] >= 100].sort_values('RPN', ascending=False)
        
        if not high_risk_items.empty:
            print("  High-risk failure modes being addressed by this AI system:")
            for _, row in high_risk_items.head(3).iterrows():
                print(f"    - {row['Potential Failure Mode']} (RPN: {row['RPN']})")
                print(f"      Mitigation: {row['Recommended Actions']}")
        
        print("=" * 60)


def main():
    """Main function to demonstrate the Dismizer system."""
    print("=== Dismizer v2.0 - Advanced Dispersion Process Optimizer ===\n")
    
    optimizer = Dismizer()

    # Add sample data if needed
    if len(optimizer.data) < 15:
        print("Adding enhanced sample data...")
        sample_data = [
            (1000, 500, 'Mixer-A', 800, 60, 15000, 25),
            (1200, 600, 'Mixer-A', 900, 70, 18000, 30),
            (3000, 800, 'Mixer-B', 500, 90, 25000, 25),
            (3500, 900, 'Mixer-B', 600, 100, 28000, 28),
            (800, 400, 'Mixer-A', 700, 50, 12000, 22),
            (5000, 1200, 'Mixer-C', 300, 150, 40000, 35),
            (5500, 1300, 'Mixer-C', 350, 160, 45000, 40),
            (2500, 700, 'Mixer-B', 450, 80, 22000, 26),
            (1500, 550, 'Mixer-A', 1000, 65, 20000, 24),
            (4800, 1100, 'Mixer-C', 280, 140, 38000, 32),
            (2200, 650, 'Mixer-B', 520, 85, 24000, 27),
            (1800, 580, 'Mixer-A', 850, 68, 19000, 29),
            (4200, 950, 'Mixer-C', 320, 145, 42000, 36),
            (2800, 720, 'Mixer-B', 480, 95, 26000, 31),
            (6000, 1400, 'Mixer-C', 260, 170, 48000, 38)
        ]
        
        for data_point in sample_data:
            optimizer.add_data(*data_point)
        print("Sample data added successfully.\n")

    # Train models
    success = optimizer.train_models()
    if not success:
        print("Model training failed. Please check your data.")
        return

    # Show equipment statistics
    stats = optimizer.get_equipment_statistics()
    if stats:
        print("\n--- Equipment Statistics ---")
        for equipment, stat in stats.items():
            print(f"{equipment}:")
            print(f"  Batches: {stat['count']}")
            print(f"  Avg RPM: {stat['avg_rpm']:.0f}")
            print(f"  Avg Time: {stat['avg_time']:.1f} min")
            print(f"  Avg Viscosity Change: {stat['avg_viscosity_change']:.0f} cP")
        print("---------------------------\n")

    # Example optimization scenarios
    test_scenarios = [
        {"initial_viscosity": 2800, "weight": 750, "temperature": 25},
        {"initial_viscosity": 1500, "weight": 600, "temperature": 30},
        {"initial_viscosity": 4500, "weight": 1000, "temperature": 35}
    ]
    
    print("=== Testing Optimization Scenarios ===")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        recommendation = optimizer.recommend_optimized_process(**scenario)
        
        if recommendation:
            print(f"\nInput: Viscosity={scenario['initial_viscosity']} cP, "
                  f"Weight={scenario['weight']} kg, Temperature={scenario['temperature']}°C")
            print("----------------------------------------------------------")
            print(f"✓ Recommended Equipment: {recommendation['equipment']}")
            print(f"✓ Recommended RPM: {recommendation['rpm']} RPM")
            print(f"✓ Estimated Time: {recommendation['estimated_time']} min")
            print(f"✓ Estimated Final Viscosity: {recommendation['estimated_viscosity']} cP")
            print(f"✓ Estimated Energy Cost: {recommendation['estimated_energy_cost']}")
            print(f"✓ Efficiency Score: {recommendation['efficiency_score']}")
            
            # FMEA Analysis
            optimizer.analyze_recommendation_with_fmea(recommendation)
        else:
            print("Failed to generate recommendation for this scenario.")
    
    print("\n=== System Ready for Production Use ===")
    print("Available methods:")
    print("- optimizer.add_data(initial_viscosity, weight, equipment, rpm, time, final_viscosity, temperature)")
    print("- optimizer.train_models()")
    print("- optimizer.recommend_optimized_process(initial_viscosity, weight, temperature)")
    print("- optimizer.get_equipment_statistics()")


if __name__ == "__main__":
    main()
