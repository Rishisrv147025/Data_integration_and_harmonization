from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

def harmonize_data(dfs):
    """Harmonizes multiple DataFrames into a single DataFrame."""
    for df in dfs:
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]  # Standardize column names
    combined_data = pd.concat(dfs, ignore_index=True)  # Concatenate DataFrames
    return combined_data

def process_data(data):
    """Fill missing values and generate summary statistics."""
    # Convert categorical columns to numeric where possible
    for col in data.select_dtypes(include=['object']):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove infinite values and replace them with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Interpolate on numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit_direction='both')
    data.fillna(method='ffill', inplace=True)  # Fill remaining NaNs using forward fill
    
    # Drop columns that are all NaN after processing
    data.dropna(axis=1, how='all', inplace=True)

    summary_statistics = data.describe(include='all')  # Generate summary statistics
    return data, summary_statistics

def preprocess_data(data):
    """Preprocesses data: scales numerical features and encodes categorical features."""
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:  # Ensure there are numeric columns to scale
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        if data[col].dtype == 'object' or data[col].dtype == 'category':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))  # Convert all values to strings
            label_encoders[col] = le  # Save the encoder for potential inverse transformations
    return data, label_encoders

def train_models(data, target_column, is_classification):
    """Train machine learning models and return their performance metrics."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target variable

    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input features contain NaN or infinite values.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_classification:
        models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            preds = model.predict(X_test)  # Make predictions
            results[name] = {
                'Accuracy': accuracy_score(y_test, preds),
                'Precision': precision_score(y_test, preds, average='weighted'),
                'Recall': recall_score(y_test, preds, average='weighted'),
                'F1 Score': f1_score(y_test, preds, average='weighted'),
            }
    else:
        models = {
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            preds = model.predict(X_test)  # Make predictions
            results[name] = {
                'MAE': mean_absolute_error(y_test, preds),
                'MSE': mean_squared_error(y_test, preds),
                'R2 Score': r2_score(y_test, preds),
            }
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    dataframes = []
    for file in files:
        if file:
            try:
                data = pd.read_csv(file)
                if data.empty:
                    return "Uploaded file is empty.", 400
                dataframes.append(data)
            except Exception as e:
                return f"Error processing file {file.filename}: {e}", 400

    # Data integration and harmonization
    integrated_data = harmonize_data(dataframes)
    
    # Data processing
    processed_data, summary_statistics = process_data(integrated_data)

    # Preprocess the data
    processed_data, label_encoders = preprocess_data(processed_data)

    # Generate initial plots
    univariate_plots = generate_univariate_plots(processed_data)
    bivariate_plots = generate_bivariate_plots(processed_data)
    correlation_matrix_plot = generate_correlation_matrix(processed_data)

    first_five_rows = processed_data.head().to_html(classes='table table-striped', index=False)

    columns = processed_data.columns.tolist()
    return render_template('visualize.html', 
                           columns=columns, 
                           data=processed_data.to_json(orient='records'),
                           first_five_rows=first_five_rows, 
                           univariate_plots=univariate_plots, 
                           bivariate_plots=bivariate_plots,
                           correlation_matrix_plot=correlation_matrix_plot)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Get JSON data from the request
    target_column = data.get('target_column')
    df_data = data.get('data')  # This should be a list of records
    is_classification = data.get('is_classification')  # Determine if classification or regression

    try:
        data_frame = pd.DataFrame.from_records(df_data)  # Create DataFrame from records
    except ValueError as e:
        return f"Error reading data: {e}", 400

    # Check if the target column exists
    if target_column not in data_frame.columns:
        return f"Target column '{target_column}' not found in the dataset.", 400

    # Train models
    model_results = train_models(data_frame, target_column, is_classification)

    return render_template('results.html', model_results=model_results, summary_statistics=data_frame.describe(include='all'))

def generate_univariate_plots(data):
    """Generate univariate plots for each numeric column."""
    plots = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        if data[column].notna().any():  # Check if the column has any non-NaN values
            plots[column] = px.histogram(data, x=column, title=f'Histogram of {column}').to_html(full_html=False)
            plots[f'box_{column}'] = px.box(data, y=column, title=f'Box Plot of {column}').to_html(full_html=False)
    return plots

def generate_bivariate_plots(data):
    """Generate bivariate plots for numeric columns."""
    plots = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if data[numeric_cols[i]].notna().any() and data[numeric_cols[j]].notna().any():
                plots[f'scatter_{i}_{j}'] = px.scatter(data, x=numeric_cols[i], y=numeric_cols[j], 
                                                        title=f'Scatter Plot of {numeric_cols[i]} vs {numeric_cols[j]}').to_html(full_html=False)

                # Line plot
                plots[f'line_{i}_{j}'] = go.Figure()
                plots[f'line_{i}_{j}'].add_trace(go.Scatter(x=data[numeric_cols[i]], y=data[numeric_cols[j]], mode='lines+markers'))
                plots[f'line_{i}_{j}'].update_layout(title=f'Line Plot of {numeric_cols[i]} vs {numeric_cols[j]}',
                                                      xaxis_title=numeric_cols[i], yaxis_title=numeric_cols[j])
                plots[f'line_{i}_{j}'] = plots[f'line_{i}_{j}'].to_html(full_html=False)

                # Bivariate density heatmap
                bivariate_density = np.histogram2d(data[numeric_cols[i]].dropna(), data[numeric_cols[j]].dropna(), bins=30)
                heatmap = go.Figure(data=go.Heatmap(z=bivariate_density[0], x=bivariate_density[1][:-1], y=bivariate_density[2][:-1],
                                                     colorscale='Viridis', colorbar=dict(title='Count')))
                heatmap.update_layout(title=f'Bivariate Density Heatmap of {numeric_cols[i]} and {numeric_cols[j]}',
                                      xaxis_title=numeric_cols[i], yaxis_title=numeric_cols[j])
                plots[f'heatmap_{i}_{j}'] = heatmap.to_html(full_html=False)
    
    return plots

def generate_correlation_matrix(data):
    """Generate a correlation matrix plot."""
    corr = data.corr()
    if corr.empty:
        return "No correlation matrix available due to insufficient numeric data."
    return px.imshow(corr, text_auto=True, title='Correlation Matrix').to_html(full_html=False)

if __name__ == '__main__':
    app.run(debug=True)
