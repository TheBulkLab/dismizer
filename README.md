# Dismizer 🔬

**Advanced Dispersion Process Optimization System**

An AI-powered system that learns from chemical dispersion process data to recommend optimal equipment settings, RPM, and processing parameters while minimizing energy costs and providing comprehensive FMEA-based risk analysis.

## ✨ Key Features

- **Equipment-Specific Modeling**: Individual machine learning models for each dispersion equipment
- **Multi-Parameter Optimization**: Optimizes RPM, processing time, and equipment selection simultaneously
- **Energy Cost Minimization**: Calculates and minimizes energy consumption
- **FMEA Integration**: Built-in Failure Mode and Effects Analysis with Control Plan
- **Model Validation**: Cross-validation and performance metrics for reliable predictions
- **Data Quality Control**: Automated outlier detection and data validation
- **Temperature Compensation**: Accounts for temperature effects on dispersion processes

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/r0bin-kim/dismizer.git
cd dismizer
pip install -r requirements.txt
```

### Basic Usage

```python
from dismizer import Dismizer

# Initialize the system
optimizer = Dismizer()

# Add process data
optimizer.add_data(
    initial_viscosity=2800,  # cP
    weight=750,              # kg
    equipment='Mixer-B',
    rpm=500,
    agitation_time=90,       # minutes
    final_viscosity=25000,   # cP
    temperature=25           # °C
)

# Train the models
optimizer.train_models()

# Get optimization recommendation
recommendation = optimizer.recommend_optimized_process(
    initial_viscosity=2800,
    weight=750,
    temperature=25
)

print(f"Recommended Equipment: {recommendation['equipment']}")
print(f"Recommended RPM: {recommendation['rpm']}")
print(f"Estimated Time: {recommendation['estimated_time']} min")
print(f"Expected Final Viscosity: {recommendation['estimated_viscosity']} cP")
```

## 📊 System Architecture

### Machine Learning Models

1. **Equipment Selector Model**: Recommends the most suitable equipment based on material properties
2. **Equipment-Specific RPM Models**: Predicts optimal RPM for each equipment type
3. **Time Prediction Models**: Estimates required processing time
4. **Viscosity Prediction Models**: Forecasts final product viscosity

### Data Flow

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Optimization → Recommendation
    ↓            ↓              ↓                 ↓              ↓            ↓
Validation → Outlier Detection → Scaling → Cross-Validation → FMEA Analysis → Output
```

## 🔧 Configuration

### Supported Equipment Types
- Mixer-A: High-speed, low-viscosity applications
- Mixer-B: Medium-speed, general purpose
- Mixer-C: Low-speed, high-viscosity applications

### Parameter Ranges
- **Initial Viscosity**: 10 - 100,000 cP
- **Weight**: 50 - 5,000 kg
- **RPM**: 100 - 2,000
- **Temperature**: 10 - 80°C
- **Processing Time**: 5 - 300 minutes

## 📈 Performance Metrics

The system provides comprehensive performance metrics:

- **RMSE** (Root Mean Square Error) for regression models
- **Accuracy** for classification models
- **Cross-validation scores** for model reliability
- **Confidence intervals** for predictions

## 🛡️ FMEA & Risk Management

### Built-in Risk Analysis

- **Failure Mode Identification**: Systematic identification of potential process failures
- **Risk Priority Number (RPN)**: Quantitative risk assessment
- **Control Plan**: Comprehensive quality control measures
- **Mitigation Strategies**: AI-powered recommendations for risk reduction

### Quality Control Features

- Automated data validation
- Outlier detection and handling
- Model performance monitoring
- Process parameter verification

## 📁 Project Structure

```
dismizer/
├── dismizer.py              # Main system class
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── CONTRIBUTING.md         # Project documentation
├── CODE_OF_CONDUCT.md      # Project documentation
└── LICENSE                 # MIT License
```

## 📊 Example Results

### Optimization Example
```
Input: Initial Viscosity=2800 cP, Weight=750 kg, Temperature=25°C
Output:
✓ Recommended Equipment: Mixer-B
✓ Recommended RPM: 520 RPM
✓ Estimated Time: 85.2 min
✓ Estimated Final Viscosity: 24,500 cP
✓ Efficiency Score: 0.1245
✓ Prediction Confidence: 94.2%
```

## 🔮 Future Enhancements

- [ ] Real-time process monitoring integration
- [ ] Web-based dashboard
- [ ] IoT sensor integration
- [ ] Advanced neural network models
- [ ] Multi-site data synchronization
- [ ] Predictive maintenance features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Industrial process optimization research community
- FMEA methodology standards
- Open source machine learning libraries (scikit-learn, pandas, numpy)

## 📞 Support

- 📖 Documentation: [Wiki](https://github.com/r0bin-kim/dismizer/wiki)
- 🐛 Bug Reports: [Issues](https://github.com/r0bin-kim/dismizer/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/r0bin-kim/dismizer/discussions)

---

**Made with ❤️ for the chemical process industry**
