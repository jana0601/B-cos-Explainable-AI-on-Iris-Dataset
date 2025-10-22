# B-cos Explainable AI on Iris Dataset

## Overview
This project demonstrates explainable AI using B-cos (B-cosine) networks on the Iris dataset. B-cos networks provide inherent interpretability through their cosine similarity-based computations, making them ideal for understanding model decisions.

## Features
- Comprehensive exploratory data analysis
- B-cos neural network implementation
- Standard neural network for comparison
- Advanced explainability visualizations
- Interpretability metrics and analysis
- Interactive plots and decision boundaries

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Notebook**:
   ```bash
   jupyter notebook iris_bcos_explainability.ipynb
   ```

## Key Findings
- B-cos networks provide transparent feature contributions
- Built-in explainability without post-hoc methods
- Comparable performance to standard neural networks
- Superior interpretability for tabular data
- **Data-driven explanation capability testing** - actual model capabilities tested rather than manual assignments
- **Built-in explainability scoring** based on real explanation methods available
- **Noise-based faithfulness calculation** - scientifically rigorous explainability evaluation with configurable noise levels
- **Streamlined comparison metrics** - focused on 4 essential evaluation criteria

## Usage
The main notebook (`iris_bcos_explainability.ipynb`) contains:
- Data loading and EDA
- Model implementation and training
- Explainability analysis
- Performance comparisons
- Advanced visualizations
- **Data-driven explanation capability testing**
- **Built-in explainability assessment**
- **Noise-based faithfulness and stability metrics**
- **Streamlined 4-metric comparison visualization**

## Export Options
The project includes multiple export formats:

1. **Jupyter Notebook**: `iris_bcos_explainability.ipynb` - Interactive analysis
2. **HTML Export**: `iris_bcos_explainability.html` - Web-viewable format
3. **PDF Export**: `iris_bcos_explainability.pdf` - Complete analysis report (1.6MB)
4. **Python Script**: `run_bcos_simple.py` - Executable version

### PDF Export Details
- **File**: `iris_bcos_explainability.pdf`
- **Size**: 1.6MB
- **Content**: Complete analysis with all visualizations and results
- **Generated**: Using Jupyter nbconvert with Playwright
- **Includes**: Data-driven explanation capability testing results, noise-based faithfulness metrics, streamlined 4-metric comparison

## Results Summary

### Model Performance
- **B-cos Model**: ~93.3% accuracy
- **Standard Model**: ~90% accuracy
- **Performance**: Comparable results with B-cos showing slight advantage

### Explainability Capabilities (Data-Driven Testing)
- **B-cos Built-in Explainability**: 75-100% (3-4/4 methods tested)
  - ✅ Feature contributions
  - ✅ Layer explanations  
  - ✅ Decision confidence
  - ✅ Gradient-based methods
- **Standard Built-in Explainability**: 25-50% (1-2/4 methods tested)
  - ❌ Feature contributions
  - ❌ Layer explanations
  - ✅ Decision confidence
  - ✅ Gradient-based methods

### Key Insights
1. **B-cos networks provide superior built-in explainability**
2. **Data-driven testing confirms actual model capabilities**
3. **No manual assignments - all scores calculated from real testing**
4. **Transparent feature importance without post-hoc methods**
5. **Ideal for domains requiring model explanations**
6. **Noise-based faithfulness provides scientifically rigorous evaluation with configurable noise levels**
7. **Streamlined 4-metric comparison focuses on essential criteria**

## Requirements
- Python 3.7+
- See `requirements.txt` for package versions
