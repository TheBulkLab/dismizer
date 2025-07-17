## Dispersion Process Optimizer (Dismizer)

The **Dispersion Process Optimizer (Dismizer)** is a machine learning-powered system designed to learn from historical dispersion process data to recommend optimal equipment and operating parameters (like RPM). The primary goal is to minimize energy costs while achieving the desired product viscosity. It integrates a **Failure Mode and Effects Analysis (FMEA)** and a **Control Plan** to provide a risk assessment for its recommendations, ensuring a robust and efficient process.

## ✨ Features

  * **Optimal Equipment Recommendation**: Automatically suggests the most suitable mixer based on initial material viscosity and batch weight.
  * **Optimized Process Parameter Prediction**: Predicts the ideal RPM and agitation time required for the dispersion process.
  * **Energy & Cost Analysis**: Estimates the final viscosity, energy cost, and an overall "efficiency score" for the recommended process.
  * **Equipment-Specific Modeling**: Trains dedicated models for each piece of equipment, capturing its unique performance characteristics.
  * **Integrated FMEA & Control Plan**: Analyzes each recommendation against a built-in FMEA to highlight how the AI mitigates high-risk failure modes. It also validates the recommendation against the process Control Plan.
  * **Actionable Risk Analysis**: Identifies critical risks using Risk Priority Numbers (RPN) and explains how the AI-driven recommendations address them.
  * **Data-Driven Improvement**: The system's models can be retrained as new data is added, allowing them to adapt and improve over time.
  * **Persistent Models**: Saves trained machine learning models to disk, eliminating the need for retraining on every run.

## ⚙️ How it Works

The `Dismizer` class encapsulates the entire workflow:

1.  **Data Handling**: The system loads process data from a `dispersion_data.csv` file. The `add_data` method allows for easy addition of new batch records.
2.  **Data Preprocessing**: The `preprocess_data` method cleans the dataset by performing feature engineering (e.g., calculating `Viscosity_Ratio`) and removing statistical outliers using the Interquartile Range (IQR) method to ensure model robustness.
3.  **Model Training**:
      * A **Random Forest Classifier** is trained to function as an `equipment_selector_model`, which recommends the best equipment for a given task.
      * For each piece of equipment, a dedicated set of **Random Forest Regressors** is trained to predict `RPM`, `Agitation_Time`, and `Final_Viscosity`. This two-tiered approach allows the system to capture the specific nuances of each machine.
4.  **Model Persistence**: All trained models, including the classifier, label encoder, and equipment-specific regressors, are saved to the `models/` directory using `joblib` for reusability.
5.  **Recommendation**: The `recommend_optimized_process` method is the core of the prediction engine.
      * It takes `initial_viscosity`, `weight`, and `temperature` as input.
      * It first uses the equipment selector to identify the best mixer.
      * It then predicts a baseline RPM and iterates through a range of nearby RPM values. For each value, it predicts the corresponding time and final viscosity.
      * An `efficiency_score`, which balances the estimated energy cost against the process outcome, is calculated to determine the single best recommendation.
6.  **FMEA and Control Plan Analysis**:
      * The `generate_fmea_control_plan` method creates a standard FMEA and Control Plan for a typical dispersion process. The FMEA calculates a **Risk Priority Number (RPN)** for each potential failure mode using the formula: $RPN = \\text{Severity} \\times \\text{Occurrence} \\times \\text{Detection}$.
      * The `analyze_recommendation_with_fmea` method takes the final recommendation and cross-references it with the FMEA and Control Plan. It reports the model's prediction confidence and explicitly states which high-RPN risks (e.g., 'Insufficient Agitation', 'Wrong Equipment Selection') are being mitigated by using the AI's optimized parameters.
7.  **Reporting**: Results are presented in a series of clear, well-structured printouts, including equipment statistics, the final recommendation, and the FMEA-based risk analysis.

### Usage

Upon running the script, the system first checks for pre-trained models. If they are not found, it trains new ones using sample data.

```
=== Dismizer v2.0 - Advanced Dispersion Process Optimizer ===

Loading existing data from 'dispersion_data.csv'.
Loaded 0 records.
Adding enhanced sample data...
New data added successfully. Total records: 1
New data added successfully. Total records: 2
...
New data added successfully. Total records: 15
Sample data added successfully.

Starting model training...
Detected 0 potential outliers.
Training models for Mixer-A...
Training models for Mixer-B...
Training models for Mixer-C...
All models saved successfully!
Model training completed successfully!

--- Model Performance Metrics ---
equipment_selector_accuracy: 1.000
Mixer-A_rpm_rmse: 53.67
Mixer-A_time_rmse: 3.42
Mixer-A_viscosity_rmse: 969.41
Mixer-B_rpm_rmse: 29.35
Mixer-B_time_rmse: 5.61
Mixer-B_viscosity_rmse: 890.31
Mixer-C_rpm_rmse: 30.69
Mixer-C_time_rmse: 6.94
Mixer-C_viscosity_rmse: 1475.29
--------------------------------

--- Equipment Statistics ---
Mixer-A:
  Batches: 5
  Avg RPM: 840
  Avg Time: 62.6 min
  Avg Viscosity Change: 15400 cP
Mixer-B:
  Batches: 5
  Avg RPM: 510
  Avg Time: 88.0 min
  Avg Viscosity Change: 21800 cP
Mixer-C:
  Batches: 5
  Avg RPM: 302
  Avg Time: 153.0 min
  Avg Viscosity Change: 36400 cP
---------------------------

=== Testing Optimization Scenarios ===

--- Scenario 1 ---
[Optimization Search] Initial Viscosity: 2800 cP, Weight: 750 kg, Temperature: 25°C

Input: Viscosity=2800 cP, Weight=750 kg, Temperature=25°C
----------------------------------------------------------
✓ Recommended Equipment: Mixer-B
✓ Recommended RPM: 494 RPM
✓ Estimated Time: 93.35 min
✓ Estimated Final Viscosity: 26059 cP
✓ Estimated Energy Cost: 181.79
✓ Efficiency Score: 204.918
--- Enhanced FMEA & Control Plan Analysis ---

[Recommended Process Parameters]
  Equipment: Mixer-B
  RPM: 494
  Time: 93.35 min
  Expected Final Viscosity: 26059 cP
  Efficiency Score: 204.918

[Equipment-Specific Model Confidence]
  Model RMSE: 890.31 cP
  Prediction Confidence: 96.6%

[Control Plan Compliance Check]
  ✓ Equipment Type: Mixer-B
    Specification: AI Recommendation
  ✓ Agitation RPM: 494 RPM
    Specification: AI Recommendation ±5%
  ✓ Agitation Time: 93.35 min
    Specification: AI Recommendation ±5%

[Risk Mitigation Analysis]
  High-risk failure modes being addressed by this AI system:
    - Insufficient Agitation (Low RPM/Short Time) (RPN: 270)
      Mitigation: Use AI-Based RPM/Time Optimization
    - Weighing Error (RPN: 96)
      Mitigation: Implement Automated Weighing
================================================================================
...
(Additional scenarios will be displayed)
...
=== System Ready for Production Use ===
Available methods:
- optimizer.add_data(initial_viscosity, weight, equipment, rpm, time, final_viscosity, temperature)
- optimizer.train_models()
- optimizer.recommend_optimized_process(initial_viscosity, weight, temperature)
- optimizer.get_equipment_statistics()
```

The system provides a clear recommendation and backs it up with a risk and control analysis, making it a powerful tool for process engineers and operators.

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
