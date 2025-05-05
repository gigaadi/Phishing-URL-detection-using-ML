# Phishing Attack Detection

This project implements a machine learning-based solution to detect phishing websites by analyzing URL features. It includes a Streamlit web application for user interaction and a comprehensive analysis pipeline for training and evaluating multiple machine learning models.

## Project Overview

The project consists of three main components:

1. **Data Analysis and Model Training** (`main.py`): Loads and preprocesses the phishing dataset, trains multiple machine learning models, and evaluates their performance.
2. **Web Application** (`app.py`): A Streamlit-based interface that allows users to input a URL and receive a prediction on whether it is a phishing site.
3. **Dataset** (`phishing.csv`): Contains features extracted from URLs labeled as phishing or legitimate.

## Features

- **Dataset**: The `phishing.csv` file includes 11,054 records with 31 features such as `UsingIP`, `LongURL`, `HTTPS`, `AnchorURL`, and `WebsiteTraffic`, with a binary `class` label (1 for legitimate, -1 for phishing).
- **Exploratory Data Analysis**:
  - Visualizations including correlation heatmaps, pair plots, and pie charts to understand feature relationships and class distribution.
  - Feature importance analysis using the Gradient Boosting Classifier.
- **Machine Learning Models**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Model Evaluation**: Metrics include accuracy, F1-score, recall, and precision, with results compared across models.
- **Web Application**: A user-friendly Streamlit app that takes a URL input, extracts features using the `FeatureExtraction` class, and predicts whether the URL is safe or unsafe using a pre-trained Gradient Boosting Classifier.

## Files

- **`main.py`**: Handles data loading, preprocessing, model training, evaluation, and visualization. It also saves the trained Gradient Boosting Classifier to `model.pkl`.
- **`app.py`**: Implements the Streamlit web application for real-time phishing detection.
- **`phishing.csv`**: Dataset containing URL features and labels.
- **`model.pkl`** (generated): Pre-trained Gradient Boosting Classifier model.
- **`feature.py`** (assumed): Contains the `FeatureExtraction` class for processing URLs (not provided but referenced in `app.py`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phishing-attack-detection.git
   cd phishing-attack-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following libraries are installed:
   - `streamlit`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `pickle`

## Usage

### Running the Web Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser (typically `http://localhost:8501`).
3. Enter a URL in the input field to check if it is safe or a phishing site.

### Training and Evaluating Models

1. Run the analysis and training script:
   ```bash
   python main.py
   ```
2. This will:
   - Load and preprocess the dataset.
   - Generate visualizations (correlation heatmap, pair plots, pie chart).
   - Train and evaluate multiple machine learning models.
   - Save the trained Gradient Boosting Classifier to `model.pkl`.
   - Display a feature importance plot.

## Model Performance

The `main.py` script evaluates models using an 80-20 train-test split. Performance metrics are stored in a DataFrame and sorted by accuracy and F1-score. The Gradient Boosting Classifier is selected for the web application due to its strong performance.

## Dataset Details

- **Source**: `phishing.csv`
- **Size**: 11,054 rows, 32 columns (including `Index` which is dropped during preprocessing).
- **Features**: 30 binary or categorical features describing URL properties (e.g., presence of HTTPS, domain age, traffic rank).
- **Target**: `class` column (1 for legitimate, -1 for phishing).

## Visualizations

- **Correlation Heatmap**: Shows relationships between features.
- **Pair Plot**: Visualizes interactions between selected features (`PrefixSuffix-`, `SubDomains`, `HTTPS`, `AnchorURL`, `WebsiteTraffic`) with class labels.
- **Pie Chart**: Displays the distribution of phishing vs. legitimate URLs.
- **Feature Importance Plot**: Highlights the most influential features for the Gradient Boosting Classifier.

## Future Improvements

- Enhance the `FeatureExtraction` class to handle edge cases or additional URL features.
- Add real-time web scraping or API integration for dynamic feature extraction.
- Improve the web interface with additional visualizations or confidence scores.
- Experiment with deep learning models or ensemble methods for better accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The dataset is sourced from a publicly available repository (specific source not provided in the code).
- Built using Python, Streamlit, and scikit-learn for educational and research purposes.