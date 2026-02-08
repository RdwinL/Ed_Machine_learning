import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Machine Learning App')

st.info('Machine learning model For Crop Yield Prediction')
DATA_URL = "https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv"
with st.expander('Raw Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv')
  df
  
st.header("Data Overview")
# Expander
with st.expander("View Dataset Information", expanded=False):

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.markdown("---")

    st.subheader("First 5 Rows (Head)")
    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Last 5 Rows (Tail)")
    st.dataframe(df.tail())

    st.markdown("---")

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.markdown("---")

    st.subheader("Missing Values Count")
    missing = df.isnull().sum()
    st.dataframe(missing)

    st.markdown("---")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
  
with st.expander('X_Data'):
  st.write('**X**')
  X =df.drop('crop_yield', axis=1)
  X


with st.expander('y_data'):
  st.write('**y**')
  y = df["crop_yield"]
  y
st.header("Initial Data Visualization")
st.subheader("Variable Distributions")

with st.expander("View Histograms for All Variables"):

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        st.subheader(f"Histogram: {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
      
with st.expander("Correlation Heatmap"):

    st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
# Utility functions
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred))
    }

def corr_heatmap(df_num: pd.DataFrame):
    corr = df_num.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap (numeric columns)"
    )
    st.plotly_chart(fig, use_container_width=True)

def pred_vs_actual_plot(y_true, y_pred, title):
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "Actual Yield", "y": "Predicted Yield"},
        title=title
    )
    minv = float(min(np.min(y_true), np.min(y_pred)))
    maxv = float(max(np.max(y_true), np.max(y_pred)))
    fig.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv)
    st.plotly_chart(fig, use_container_width=True)

def residuals_plot(y_true, y_pred, title):
    res = y_true - y_pred
    fig = px.scatter(
        x=y_pred, y=res,
        labels={"x": "Predicted Yield", "y": "Residual (Actual - Predicted)"},
        title=title
    )
    fig.add_hline(y=0)
    st.plotly_chart(fig, use_container_width=True)

def feature_importance_bar(importances, feature_names, title):
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True)
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title=title)
    st.plotly_chart(fig, use_container_width=True)

def linear_coef_bar(coefs, feature_names, title):
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df = coef_df.sort_values("coefficient")
    fig = px.bar(coef_df, x="coefficient", y="feature", orientation="h", title=title)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar: Data Source
# -----------------------------
st.sidebar.header("1) Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_repo_file = st.sidebar.checkbox(f"Use repo file: {DEFAULT_DATA_PATH}", value=True)

df = None
load_error = None

try:
    if uploaded is not None:
        df = load_data_from_upload(uploaded)
    elif use_repo_file:
        df = load_data_from_path(DEFAULT_DATA_PATH)
    else:
        st.info("Upload a CSV file or enable 'Use repo file'.")
        st.stop()
except Exception as e:
    load_error = str(e)

if df is None:
    st.error(f"Failed to load data: {load_error}")
    st.stop()

# -----------------------------
# Sidebar: Target/Features
# -----------------------------
st.sidebar.header("2) Target & Features")

if DEFAULT_TARGET in df.columns:
    default_target_index = df.columns.tolist().index(DEFAULT_TARGET)
else:
    # fallback: use last column
    default_target_index = len(df.columns) - 1

target_col = st.sidebar.selectbox(
    "Select target column (Yield)",
    options=df.columns.tolist(),
    index=default_target_index
)

# Numeric columns only (this dataset is numeric, but keep robust)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if target_col not in numeric_cols:
    st.error("Target column must be numeric for regression.")
    st.stop()

default_features = [c for c in numeric_cols if c != target_col]
feature_cols = st.sidebar.multiselect(
    "Select feature columns",
    options=numeric_cols,
    default=default_features
)

if len(feature_cols) == 0:
    st.warning("Select at least one feature column.")
    st.stop()

# -----------------------------
# Sidebar: Modeling controls
# -----------------------------
st.sidebar.header("3) Training Settings")

test_size = st.sidebar.slider("Test split size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, 1)

st.sidebar.subheader("Decision Tree Hyperparameters")
dt_max_depth = st.sidebar.slider("max_depth", 1, 30, 8)
dt_min_samples_split = st.sidebar.slider("min_samples_split", 2, 30, 2)
dt_min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 30, 1)

use_rf = st.sidebar.checkbox("Also train Random Forest (advanced)", value=True)
rf_estimators = st.sidebar.slider("RF n_estimators", 50, 500, 200, 50) if use_rf else 0
rf_max_depth = st.sidebar.slider("RF max_depth", 2, 50, 15) if use_rf else None

do_cv = st.sidebar.checkbox("Add 5-fold cross validation (R²)", value=True)

# -----------------------------
# Tabs
# -----------------------------
tab_data, tab_eda, tab_train, tab_predict = st.tabs(
    ["📄 Data", "📊 EDA & Visualizations", "🧠 Train & Evaluate", "🧮 Predict"]
)

# -----------------------------
# Tab 1: Data
# -----------------------------
with tab_data:
    st.subheader("Dataset Preview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(25), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Data types")
        st.write(df.dtypes.astype(str))
    with c2:
        st.markdown("### Missing values")
        st.write(df.isna().sum())
    with c3:
        st.markdown("### Summary (selected)")
        st.write(df[feature_cols + [target_col]].describe())

    # Download selected subset
    subset = df[feature_cols + [target_col]].dropna().copy()
    st.download_button(
        "⬇️ Download selected clean data (CSV)",
        data=subset.to_csv(index=False).encode("utf-8"),
        file_name="crop_yield_selected_clean.csv",
        mime="text/csv"
    )

# -----------------------------
# Tab 2: EDA
# -----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    work = df[feature_cols + [target_col]].copy()

    # Column chooser
    col_for_dist = st.selectbox("Pick a column for distribution plots", options=feature_cols + [target_col])

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(work, x=col_for_dist, nbins=30, title=f"Histogram: {col_for_dist}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(work, y=col_for_dist, title=f"Boxplot: {col_for_dist}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Feature vs Target Relationship")
    x_feat = st.selectbox("Choose feature (x-axis)", options=feature_cols)
    fig_sc = px.scatter(
        work,
        x=x_feat,
        y=target_col,
        title=f"{x_feat} vs {target_col}"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("### Pairwise Scatter (sampled for speed)")
    sample_n = min(500, len(work.dropna()))
    sampled = work.dropna().sample(sample_n, random_state=int(random_state))
    fig_matrix = px.scatter_matrix(
        sampled,
        dimensions=feature_cols + [target_col],
        title="Scatter Matrix (sampled)"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("### Correlation Heatmap")
    corr_heatmap(work.dropna())

# -----------------------------
# Tab 3: Train & Evaluate
# -----------------------------
with tab_train:
    st.subheader("Train Models + Evaluate + Compare")

    # Prepare clean data
    data = df[feature_cols + [target_col]].dropna().copy()
    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state)
    )

    # Models
    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

    dt = DecisionTreeRegressor(
        max_depth=int(dt_max_depth),
        min_samples_split=int(dt_min_samples_split),
        min_samples_leaf=int(dt_min_samples_leaf),
        random_state=int(random_state)
    )

    rf = None
    if use_rf:
        rf = RandomForestRegressor(
            n_estimators=int(rf_estimators),
            max_depth=int(rf_max_depth),
            random_state=int(random_state),
            n_jobs=-1
        )

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training..."):
            # Train LR
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_metrics = evaluate(y_test, lr_pred)

            # Train DT
            dt.fit(X_train, y_train)
            dt_pred = dt.predict(X_test)
            dt_metrics = evaluate(y_test, dt_pred)

            results = [
                {"Model": "Linear Regression", **lr_metrics},
                {"Model": "Decision Tree", **dt_metrics},
            ]

            preds = {
                "Linear Regression": lr_pred,
                "Decision Tree": dt_pred
            }

            models = {
                "Linear Regression": lr,
                "Decision Tree": dt
            }

            # Train RF
            if rf is not None:
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_metrics = evaluate(y_test, rf_pred)
                results.append({"Model": "Random Forest", **rf_metrics})
                preds["Random Forest"] = rf_pred
                models["Random Forest"] = rf

            results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

            # Optional CV
            if do_cv:
                kf = KFold(n_splits=5, shuffle=True, random_state=int(random_state))
                results_df["CV_R2_Mean"] = np.nan

                lr_cv = cross_val_score(lr, X, y, cv=kf, scoring="r2").mean()
                dt_cv = cross_val_score(dt, X, y, cv=kf, scoring="r2").mean()

                results_df.loc[results_df["Model"] == "Linear Regression", "CV_R2_Mean"] = float(lr_cv)
                results_df.loc[results_df["Model"] == "Decision Tree", "CV_R2_Mean"] = float(dt_cv)

                if rf is not None:
                    rf_cv = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
                    results_df.loc[results_df["Model"] == "Random Forest", "CV_R2_Mean"] = float(rf_cv)

            best_model_name = results_df.loc[0, "Model"]
            best_model = models[best_model_name]

            # Store in session
            st.session_state["models"] = models
            st.session_state["preds"] = preds
            st.session_state["results_df"] = results_df
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_model"] = best_model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

        st.success(f"Training complete ✅ Best model: **{best_model_name}**")

    # Display results if trained
    if "results_df" in st.session_state:
        st.markdown("### Model Performance Comparison")
        st.dataframe(st.session_state["results_df"], use_container_width=True)

        fig_r2 = px.bar(st.session_state["results_df"], x="Model", y="R2", title="R² Score Comparison (higher is better)")
        st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("### Model Diagnostics")
        model_choice = st.selectbox("Choose model for plots", options=list(st.session_state["preds"].keys()))
        y_true = st.session_state["y_test"]
        y_pred = st.session_state["preds"][model_choice]

        c1, c2 = st.columns(2)
        with c1:
            pred_vs_actual_plot(y_true, y_pred, f"{model_choice}: Predicted vs Actual")
        with c2:
            residuals_plot(y_true, y_pred, f"{model_choice}: Residuals (Actual - Predicted)")

        st.markdown("### Model Interpretation")
        chosen_model = st.session_state["models"][model_choice]

        if model_choice == "Linear Regression":
            # pipeline: scaler + model
            lr_model = chosen_model.named_steps["model"]
            linear_coef_bar(lr_model.coef_, feature_cols, "Linear Regression Coefficients")
        else:
            if hasattr(chosen_model, "feature_importances_"):
                feature_importance_bar(chosen_model.feature_importances_, feature_cols, f"{model_choice} Feature Importance")

        st.markdown("### Export Best Model")
        best_name = st.session_state["best_model_name"]
        best_model = st.session_state["best_model"]

        bundle = {
            "model_name": best_name,
            "model": best_model,
            "features": feature_cols,
            "target": target_col
        }

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            f"⬇️ Download Best Model ({best_name})",
            data=buf.getvalue(),
            file_name="best_crop_yield_model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.info("Click **Train Models** to see evaluation tables and model visualizations.")

# -----------------------------
# Tab 4: Predict
# -----------------------------
with tab_predict:
    st.subheader("Interactive Prediction")

    st.write("Provide values for each feature and predict crop yield using the **best trained model**.")

    if "best_model" not in st.session_state:
        st.warning("Train models first in the **Train & Evaluate** tab.")
    else:
        # Build input form from feature ranges
        inputs = {}
        cols = st.columns(3)

        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_mean = float(df[col].mean())
                inputs[col] = st.number_input(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=col_mean
                )

        if st.button("🌾 Predict Yield", type="primary"):
            X_in = pd.DataFrame([inputs], columns=feature_cols)
            model = st.session_state["best_model"]
            model_name = st.session_state["best_model_name"]

            pred = float(model.predict(X_in)[0])
            st.success(f"Predicted **{target_col}**: **{pred:.2f}**  (Model: **{model_name}**)")

        st.caption("Tip: For Streamlit Cloud deployment, keep `crop_yield_data.csv` inside your GitHub repo.")
