import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import shap
from sklearn.model_selection import GridSearchCV

# Harmonize data by standardizing column names and combining multiple dataframes
def harmonize_data(dfs):
    for df in dfs:
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]  # Standardize column names
    return pd.concat(dfs, ignore_index=True)

# Process data by handling missing values, converting categorical data to numeric, and generating summary statistics
def process_data(data):
    for col in data.select_dtypes(include=['object']):
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, if possible
    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    data.interpolate(method='linear', limit_direction='both', inplace=True)  # Interpolate missing numeric values
    data.fillna(method='ffill', inplace=True)  # Forward fill remaining NaNs
    data.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values
    return data, data.describe(include='all')  # Return processed data and summary statistics

# Preprocess data by scaling numeric features and encoding categorical features
def preprocess_data(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        if data[col].dtype in ['object', 'category']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))  # Convert to string before encoding
            label_encoders[col] = le
    return data, label_encoders

# Train machine learning models and return their performance metrics
def train_models(data, target_column, is_classification, hyperparameter_tuning=False):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Decision Tree': DecisionTreeClassifier() if is_classification else DecisionTreeRegressor(),
        'Random Forest': RandomForestClassifier() if is_classification else RandomForestRegressor(),
        'XGBoost': XGBClassifier() if is_classification else XGBRegressor(),
    }

    # Hyperparameter tuning with GridSearchCV
    if hyperparameter_tuning:
        param_grid = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            },
            'XGBoost': {
                'learning_rate': [0.01, 0.1],
                'n_estimators': [50, 100, 200],
            }
        }
        
        results = {}
        for model_name, model in models.items():
            if model_name in param_grid:
                grid_search = GridSearchCV(model, param_grid[model_name], cv=3)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                preds = best_model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            results[model_name] = evaluate_model(y_test, preds, is_classification)
        return results
    
    else:
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = evaluate_model(y_test, preds, is_classification)
        
        return results

# Evaluate model performance and return metrics
def evaluate_model(y_test, preds, is_classification):
    if is_classification:
        return {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted'),
            'Recall': recall_score(y_test, preds, average='weighted'),
            'F1 Score': f1_score(y_test, preds, average='weighted'),
        }
    else:
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'R2 Score': r2_score(y_test, preds),
        }

# Generate univariate plots (histograms and box plots) for numeric columns
def generate_univariate_plots(data):
    plots = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        if data[column].notna().any():  # Skip columns with only NaN values
            plots[column] = px.histogram(data, x=column, title=f'Histogram of {column}')
            plots[f'box_{column}'] = px.box(data, y=column, title=f'Box Plot of {column}')
    return plots

# Generate bivariate plots (scatter plots, line plots, heatmaps) for pairs of numeric columns
def generate_bivariate_plots(data):
    plots = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if data[numeric_cols[i]].notna().any() and data[numeric_cols[j]].notna().any():
                plots[f'scatter_{i}_{j}'] = px.scatter(data, x=numeric_cols[i], y=numeric_cols[j], 
                                                        title=f'Scatter Plot of {numeric_cols[i]} vs {numeric_cols[j]}')
                plots[f'line_{i}_{j}'] = go.Figure()
                plots[f'line_{i}_{j}'].add_trace(go.Scatter(x=data[numeric_cols[i]], y=data[numeric_cols[j]], mode='lines+markers'))
                plots[f'line_{i}_{j}'].update_layout(title=f'Line Plot of {numeric_cols[i]} vs {numeric_cols[j]}',
                                                      xaxis_title=numeric_cols[i], yaxis_title=numeric_cols[j])
                plots[f'line_{i}_{j}'] = plots[f'line_{i}_{j}']
                bivariate_density = np.histogram2d(data[numeric_cols[i]].dropna(), data[numeric_cols[j]].dropna(), bins=30)
                heatmap = go.Figure(data=go.Heatmap(z=bivariate_density[0], x=bivariate_density[1][:-1], y=bivariate_density[2][:-1], colorscale='Viridis'))
                heatmap.update_layout(title=f'Bivariate Density Heatmap of {numeric_cols[i]} and {numeric_cols[j]}',
                                      xaxis_title=numeric_cols[i], yaxis_title=numeric_cols[j])
                plots[f'heatmap_{i}_{j}'] = heatmap
    return plots

# Generate a correlation matrix plot
def generate_correlation_matrix(data):
    corr = data.corr()
    if corr.empty:
        return "No correlation matrix available due to insufficient numeric data."
    return px.imshow(corr, text_auto=True, title='Correlation Matrix')

# Streamlit interface
def main():
    st.title("Data Science Workflow App")

    # File Upload
    st.sidebar.header("Upload Data")
    uploaded_files = st.sidebar.file_uploader("Choose Files", accept_multiple_files=True, type=['csv', 'xlsx', 'json', 'parquet'])

    if uploaded_files:
        dataframes = []
        st.subheader("Uploaded Datasets")
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith('xlsx'):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('json'):
                    data = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('parquet'):
                    data = pd.read_parquet(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)
                    
                if data.empty:
                    st.error(f"Uploaded file {uploaded_file.name} is empty.")
                dataframes.append(data)
                st.write(f"**Dataset Name:** {uploaded_file.name}")
                st.write(data.head())  # Display first few rows of the dataset
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")

        # Data Integration and Harmonization
        st.subheader("Data Integration and Harmonization")
        integrated_data = harmonize_data(dataframes)
        st.write("Data successfully integrated and harmonized.")
        st.write(integrated_data.head())  # Show the first few rows of the combined data

        processed_data, summary_statistics = process_data(integrated_data)

        # Preprocessing
        processed_data, label_encoders = preprocess_data(processed_data)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(summary_statistics)

        # Generate Plots
        st.subheader("Univariate Plots")
        univariate_plots = generate_univariate_plots(processed_data)
        for column, plot in univariate_plots.items():
            st.plotly_chart(plot)

        st.subheader("Bivariate Plots")
        bivariate_plots = generate_bivariate_plots(processed_data)
        for key, plot in bivariate_plots.items():
            st.plotly_chart(plot)

        st.subheader("Correlation Matrix")
        correlation_matrix_plot = generate_correlation_matrix(processed_data)
        st.plotly_chart(correlation_matrix_plot)

        # Model Training
        st.sidebar.header("Model Training")
        target_column = st.sidebar.selectbox("Select Target Column", processed_data.columns.tolist())

        if target_column:
            is_classification = st.sidebar.radio("Is this a classification task?", ["Yes", "No"]) == "Yes"
            hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning")

            if st.sidebar.button("Train Models"):
                model_results = train_models(processed_data, target_column, is_classification, hyperparameter_tuning)
                st.subheader("Model Performance")
                st.write(model_results)

if __name__ == '__main__':
    main()
