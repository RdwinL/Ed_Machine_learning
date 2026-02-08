import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import io

# Page configuration
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")

st.title('🌾 Machine Learning App')
st.info('Machine learning model for Crop Yield Prediction')

# Load data
DATA_URL = "https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Data Overview",
    "Data Visualization", 
    "Model Training",
    "Model Evaluation",
    "Make Predictions"
])

# ==================== DATA OVERVIEW PAGE ====================
if page == "Data Overview":
    st.header("📊 Data Overview")
    
    with st.expander('Raw Data', expanded=True):
        st.write('**Raw data**')
        st.dataframe(df, use_container_width=True)
    
    # Dataset Information
    with st.expander("View Dataset Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            
            st.markdown("---")
            st.subheader("Data Types")
            st.write(df.dtypes)
        
        with col2:
            st.subheader("Missing Values Count")
            missing = df.isnull().sum()
            st.dataframe(missing)
            
            st.markdown("---")
            st.subheader("Duplicate Rows")
            st.write(f"Number of duplicates: {df.duplicated().sum()}")
        
        st.markdown("---")
        st.subheader("First 5 Rows (Head)")
        st.dataframe(df.head())
        
        st.markdown("---")
        st.subheader("Last 5 Rows (Tail)")
        st.dataframe(df.tail())
        
        st.markdown("---")
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
    
    # X and y data
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander('X_Data (Features)'):
            st.write('**X**')
            X = df.drop('crop_yield', axis=1)
            st.dataframe(X)
            st.write(f"Shape: {X.shape}")
    
    with col2:
        with st.expander('y_data (Target)'):
            st.write('**y**')
            y = df["crop_yield"]
            st.dataframe(y)
            st.write(f"Shape: {y.shape}")

# ==================== DATA VISUALIZATION PAGE ====================
elif page == "Data Visualization":
    st.header("📈 Data Visualization")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Distribution plots
    st.subheader("Variable Distributions")
    with st.expander("View Histograms for All Variables", expanded=True):
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.histplot(df[col_name], bins=30, kde=True, ax=ax, color='skyblue')
                    ax.set_title(f"{col_name}")
                    ax.set_xlabel(col_name)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                    plt.close()
    
    # Correlation Heatmap
    st.subheader("Correlation Analysis")
    with st.expander("Correlation Heatmap", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, 
                    center=0, linewidths=1, fmt='.2f')
        ax.set_title("Correlation Matrix Heatmap")
        st.pyplot(fig)
        plt.close()
    
    # Box plots
    st.subheader("Outlier Detection")
    with st.expander("Box Plots for All Variables", expanded=False):
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.boxplot(y=df[col_name], ax=ax, color='lightgreen')
                    ax.set_title(f"{col_name}")
                    st.pyplot(fig)
                    plt.close()
    
    # Scatter plots
    st.subheader("Feature vs Target Relationships")
    with st.expander("Scatter Plots", expanded=False):
        feature_cols = [col for col in numeric_cols if col != 'crop_yield']
        cols_per_row = 3
        for i in range(0, len(feature_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(feature_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(df[col_name], df['crop_yield'], alpha=0.5)
                    ax.set_xlabel(col_name)
                    ax.set_ylabel('crop_yield')
                    ax.set_title(f"{col_name} vs Crop Yield")
                    st.pyplot(fig)
                    plt.close()

# ==================== MODEL TRAINING PAGE ====================
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    # Prepare data
    X = df.drop('crop_yield', axis=1)
    y = df['crop_yield']
    
    # Training configuration
    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    with col2:
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "Linear Regression", "Decision Tree"]
        )
    
    # Model-specific parameters
    if model_choice == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        with col2:
            max_depth = st.slider("Max Depth", 3, 30, 10, 1)
    
    # Train button
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            # Initialize model
            if model_choice == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state
                )
            elif model_choice == "Linear Regression":
                model = LinearRegression()
            else:  # Decision Tree
                model = DecisionTreeRegressor(random_state=random_state)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_train_pred'] = y_train_pred
            st.session_state['y_test_pred'] = y_test_pred
            st.session_state['feature_names'] = X.columns.tolist()
            
            st.success("✅ Model trained successfully!")
            
            # Display metrics
            st.subheader("Training Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training R² Score", f"{train_r2:.4f}")
                st.metric("Training MSE", f"{train_mse:.4f}")
                st.metric("Training MAE", f"{train_mae:.4f}")
            
            with col2:
                st.metric("Test R² Score", f"{test_r2:.4f}")
                st.metric("Test MSE", f"{test_mse:.4f}")
                st.metric("Test MAE", f"{test_mae:.4f}")
            
            with col3:
                st.metric("Training RMSE", f"{np.sqrt(train_mse):.4f}")
                st.metric("Test RMSE", f"{np.sqrt(test_mse):.4f}")
                
                # Overfitting indicator
                r2_diff = train_r2 - test_r2
                if r2_diff > 0.1:
                    st.warning("⚠️ Possible overfitting detected")
                else:
                    st.success("✅ Good generalization")
            
            # Feature importance (for tree-based models)
            if model_choice in ["Random Forest", "Decision Tree"]:
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)
                plt.close()

# ==================== MODEL EVALUATION PAGE ====================
elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first on the 'Model Training' page.")
    else:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_train = st.session_state['y_train']
        y_train_pred = st.session_state['y_train_pred']
        y_test_pred = st.session_state['y_test_pred']
        
        # Metrics
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set**")
            train_metrics = pd.DataFrame({
                'Metric': ['R² Score', 'MSE', 'RMSE', 'MAE'],
                'Value': [
                    r2_score(y_train, y_train_pred),
                    mean_squared_error(y_train, y_train_pred),
                    np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    mean_absolute_error(y_train, y_train_pred)
                ]
            })
            st.dataframe(train_metrics, use_container_width=True)
        
        with col2:
            st.write("**Test Set**")
            test_metrics = pd.DataFrame({
                'Metric': ['R² Score', 'MSE', 'RMSE', 'MAE'],
                'Value': [
                    r2_score(y_test, y_test_pred),
                    mean_squared_error(y_test, y_test_pred),
                    np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    mean_absolute_error(y_test, y_test_pred)
                ]
            })
            st.dataframe(test_metrics, use_container_width=True)
        
        # Visualization
        st.subheader("Prediction Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
            ax.plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Crop Yield')
            ax.set_ylabel('Predicted Crop Yield')
            ax.set_title('Actual vs Predicted (Test Set)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Residual plot
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Crop Yield')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot (Test Set)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Distribution comparison
        st.subheader("Distribution Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(y_test, bins=30, alpha=0.5, label='Actual', color='blue')
        ax.hist(y_test_pred, bins=30, alpha=0.5, label='Predicted', color='orange')
        ax.set_xlabel('Crop Yield')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution: Actual vs Predicted')
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        # Error distribution
        st.subheader("Error Analysis")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors')
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        st.pyplot(fig)
        plt.close()
        
        # Download model
        st.subheader("Download Model")
        model_bytes = pickle.dumps(model)
        st.download_button(
            label="Download Trained Model",
            data=model_bytes,
            file_name="crop_yield_model.pkl",
            mime="application/octet-stream"
        )

# ==================== MAKE PREDICTIONS PAGE ====================
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first on the 'Model Training' page.")
    else:
        model = st.session_state['model']
        feature_names = st.session_state['feature_names']
        
        # Prediction method selection
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Manual Input", "Upload CSV File"]
        )
        
        if prediction_method == "Manual Input":
            st.subheader("Enter Feature Values")
            st.write("Please enter the values for each feature:")
            
            # Create input fields based on feature names
            input_data = {}
            
            # Create columns for better layout
            col1, col2, col3 = st.columns(3)
            
            for idx, feature in enumerate(feature_names):
                # Get some statistics from the original data for helpful hints
                feature_mean = df[feature].mean()
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                
                with [col1, col2, col3][idx % 3]:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(feature_mean),
                        min_value=float(feature_min),
                        max_value=float(feature_max),
                        help=f"Range: {feature_min:.2f} - {feature_max:.2f}, Mean: {feature_mean:.2f}"
                    )
            
            if st.button("Predict Crop Yield", type="primary"):
                # Create DataFrame with input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.success(f"### Predicted Crop Yield: **{prediction:.2f}**")
                
                # Show input summary
                with st.expander("View Input Summary"):
                    st.dataframe(input_df.T.rename(columns={0: 'Value'}))
        
        else:  # Upload CSV File
            st.subheader("Upload CSV File for Batch Predictions")
            st.write("Upload a CSV file with the same features as the training data.")
            
            # Show expected format
            with st.expander("View Expected CSV Format"):
                st.write("Your CSV should have the following columns:")
                st.code(", ".join(feature_names))
                
                # Show sample
                sample_df = df[feature_names].head(3)
                st.write("Example:")
                st.dataframe(sample_df)
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Read uploaded file
                    upload_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Uploaded Data:**")
                    st.dataframe(upload_df)
                    
                    # Check if all required columns are present
                    missing_cols = set(feature_names) - set(upload_df.columns)
                    extra_cols = set(upload_df.columns) - set(feature_names)
                    
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                    else:
                        # Select only the required columns in the correct order
                        upload_df_features = upload_df[feature_names]
                        
                        if st.button("Generate Predictions", type="primary"):
                            # Make predictions
                            predictions = model.predict(upload_df_features)
                            
                            # Add predictions to dataframe
                            result_df = upload_df.copy()
                            result_df['Predicted_Crop_Yield'] = predictions
                            
                            st.success("✅ Predictions generated successfully!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(result_df)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Predicted Yield", f"{predictions.mean():.2f}")
                            with col2:
                                st.metric("Min Predicted Yield", f"{predictions.min():.2f}")
                            with col3:
                                st.metric("Max Predicted Yield", f"{predictions.max():.2f}")
                            
                            # Visualization
                            st.subheader("Prediction Distribution")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
                            ax.set_xlabel('Predicted Crop Yield')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Predicted Crop Yields')
                            st.pyplot(fig)
                            plt.close()
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="crop_yield_predictions.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**App Features:**
- Data exploration and visualization
- Multiple ML models
- Model evaluation metrics
- Manual and batch predictions
""")
