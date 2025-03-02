# Time Series Analysis Framework

A Python-based framework for financial time series analysis and trading strategy development, with a focus on technical analysis and machine learning-based classification and modular design for identifying performant combinations of components and parameters.

## Project Structure

```
timeseriesanalysis/
├── classification_analysis.ipynb  # Main analysis notebook
├── code/
│   ├── allocators/              # Portfolio allocation strategies
│   ├── classifiers/             # ML model implementations
│   │   ├── abstract/           # Base classifier definitions
│   │   ├── Benchmark.py        # Benchmark model implementations
│   │   ├── Pytorch_NN.py       # Neural network models
│   │   └── Sklearn_GSCV.py     # Scikit-learn models with grid search
│   ├── datasets/               # Data preprocessing and feature engineering
│   ├── prices/                 # Price data handling and processing
│   ├── traders/               # Trading strategy implementations
│   ├── ChartFunctions.py      # Visualization utilities
│   └── RunFunctions.py        # Core execution functions
└── output/                    # Output directory for results and visualizations
```

## Features

### Technical Analysis
- Uses the `ta` library to calculate standard technical indicators

### Data Processing
- Technical indicator calculation and feature engineering
- Custom time-series transformations (shifting, differencing)
- Flexible data sampling and windowing

### Classifier Layer - Machine Learning Models
- Extensible classifier framework supporting various ML approaches
- Pre-built implementations for:
  - Scikit-learn models with grid search optimization
  - PyTorch neural networks
  - Basic benchmark models

### Trading Layer
- Rule-based trading systems driven by model signals
- Binary trading implementation for buy/sell decisions
- Modular design allowing quick implementation of new trading rules
- Performance tracking and signal analysis

### Allocation Layer
- Dynamic allocation between multiple trading strategies
- Performance-based weighting
- Custom allocation rules through modular allocator design

### Modular Architecture
- Each component (indicators, datasets, models, traders, allocators) is independently configurable
- Easy to extend with new implementations at any layer
- Mix and match different components to create custom strategies
- Built-in visualisation tools to identify high-performing component combinations

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- pytorch
- ta (Technical Analysis library)

## Usage

The main entry point is `classification_analysis.ipynb`, which demonstrates:
1. Data loading and preprocessing
2. Feature engineering and technical analysis
3. Model training and evaluation
4. Trading strategy implementation
5. Performance visualization and analysis

## Getting Started

1. Clone the repository
2. Install required dependencies
3. Open `classification_analysis.ipynb` to see example usage
4. Modify model parameters and trading strategies as needed
