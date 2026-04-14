import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression

# --- UI Config ---
st.set_page_config(page_title="Salary Predictor ML Pipeline", layout="wide")
st.title("🚀 Salary Predictor: End-to-End ML Pipeline")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Step 1: Problem Definition")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Regression"])

# --- Session State ---
if "df" not in st.session_state:
    st.session_state.df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None

# --- Tabs ---
tabs = st.tabs([
    "📥 Data Input", "🔍 EDA", "🛠 Data Engineering",
    "🎯 Feature Selection", "📊 Model Training"
])

# =========================
# TAB 1: DATA INPUT
# =========================
with tabs[0]:
    st.header("1. Data Input & Visualization")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("### Data Preview")
            st.dataframe(df.head())

            target_col = st.selectbox("Select Target Column", df.columns.tolist())
            st.session_state.target_col = target_col

            # PCA
            st.subheader("PCA Visualization")
            numeric_df = df.select_dtypes(include=[np.number]).dropna()

            if len(numeric_df.columns) >= 2:
                features = st.multiselect(
                    "Select Features for PCA",
                    numeric_df.columns.tolist(),
                    default=numeric_df.columns[:2].tolist(),
                )

                if len(features) >= 2:
                    try:
                        pca = PCA(n_components=2)
                        scaled = StandardScaler().fit_transform(numeric_df[features])
                        components = pca.fit_transform(scaled)
                        fig = px.scatter(
                            x=components[:, 0],
                            y=components[:, 1],
                            labels={"x": "PC1", "y": "PC2"},
                            title="PCA Plot",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"PCA failed: {e}")
            else:
                st.info("Need at least 2 numeric columns for PCA.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Upload a CSV file to get started.")

# =========================
# TAB 2: EDA
# =========================
with tabs[1]:
    st.header("2. Exploratory Data Analysis")

    if st.session_state.df is None:
        st.info("Please upload a CSV file in the Data Input tab first.")
    else:
        try:
            df = st.session_state.df

            st.write("### Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.shape[1] >= 2:
                # FIX: drop columns that are all-NaN before computing corr
                corr = numeric_df.dropna(axis=1, how="all").corr()
                # FIX: fill remaining NaN in corr matrix so imshow doesn't crash
                corr = corr.fillna(0)
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation matrix.")

            if st.session_state.target_col and st.session_state.target_col in df.columns:
                st.write("### Target Distribution")
                fig_hist = px.histogram(
                    df, x=st.session_state.target_col,
                    title=f"Distribution of {st.session_state.target_col}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"EDA error: {e}")

# =========================
# TAB 3: DATA ENGINEERING
# =========================
with tabs[2]:
    st.header("3. Data Cleaning & Outliers")

    if st.session_state.df is None:
        st.info("Please upload a CSV file in the Data Input tab first.")
    else:
        try:
            df = st.session_state.df.copy()

            st.write(f"**Current shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            # Imputation
            method = st.selectbox("Imputation Strategy", ["Mean", "Median"])

            if st.button("Apply Imputation"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    # FIX: avoid inplace on copy — assign directly
                    if method == "Mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].median())
                st.session_state.df = df
                st.success(f"Imputation applied using {method}. Null counts reduced.")
                st.dataframe(df.isnull().sum().rename("Nulls Remaining"))

            # Outliers
            if st.button("Remove Outliers (Isolation Forest)"):
                numeric = df.select_dtypes(include=[np.number]).dropna()
                if numeric.shape[0] > 10:
                    iso = IsolationForest(contamination=0.1, random_state=42)
                    preds = iso.fit_predict(numeric)
                    # FIX: align index — numeric may have dropped rows
                    mask = pd.Series(preds, index=numeric.index) != -1
                    df = df.loc[mask]
                    st.session_state.df = df
                    removed = (~mask).sum()
                    st.success(f"Outliers removed: {removed} rows. New shape: {df.shape}")
                else:
                    st.warning("Not enough rows to run Isolation Forest.")
        except Exception as e:
            st.error(f"Data engineering error: {e}")

# =========================
# TAB 4: FEATURE SELECTION
# =========================
with tabs[3]:
    st.header("4. Feature Selection")

    if st.session_state.df is None:
        st.info("Please upload a CSV file in the Data Input tab first.")
    elif not st.session_state.target_col:
        st.info("Please select a target column in the Data Input tab.")
    else:
        try:
            df = st.session_state.df
            target = st.session_state.target_col

            if target not in df.columns:
                st.warning(f"Target column '{target}' not found in current dataframe.")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                if target not in numeric_cols:
                    st.error("Target column must be numeric for regression.")
                else:
                    feature_cols = [c for c in numeric_cols if c != target]

                    if not feature_cols:
                        st.warning("No numeric feature columns available.")
                    else:
                        # FIX: drop rows with NaN before mutual info
                        clean = df[feature_cols + [target]].dropna()
                        X = clean[feature_cols]
                        y = clean[target]

                        scores = mutual_info_regression(X, y, random_state=42)
                        importance = (
                            pd.Series(scores, index=feature_cols)
                            .sort_values(ascending=False)
                            .reset_index()
                        )
                        importance.columns = ["Feature", "Mutual Info Score"]

                        fig = px.bar(
                            importance, x="Feature", y="Mutual Info Score",
                            title="Feature Importance (Mutual Information)",
                            color="Mutual Info Score", color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(importance)
        except Exception as e:
            st.error(f"Feature selection error: {e}")

# =========================
# TAB 5: MODEL TRAINING
# =========================
with tabs[4]:
    st.header("5. Model Training")

    if st.session_state.df is None:
        st.info("Please upload a CSV file in the Data Input tab first.")
    elif not st.session_state.target_col:
        st.info("Please select a target column in the Data Input tab.")
    else:
        try:
            df = st.session_state.df
            target = st.session_state.target_col

            if target not in df.columns:
                st.warning(f"Target column '{target}' not found in current dataframe.")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [c for c in numeric_cols if c != target]

                if not feature_cols:
                    st.warning("No numeric feature columns available.")
                else:
                    # FIX: drop NaN rows before training
                    clean = df[feature_cols + [target]].dropna()

                    if clean.shape[0] < 10:
                        st.warning("Not enough clean rows to train a model (need at least 10).")
                    else:
                        X = clean[feature_cols]
                        y = clean[target]

                        model_choice = st.selectbox(
                            "Select Model",
                            ["Linear Regression", "Random Forest", "SVM"],
                        )

                        if st.button("Train Model"):
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )

                            with st.spinner("Training..."):
                                if model_choice == "Linear Regression":
                                    model = LinearRegression()
                                elif model_choice == "Random Forest":
                                    model = RandomForestRegressor(random_state=42)
                                else:
                                    model = SVR()

                                model.fit(X_train, y_train)
                                score = model.score(X_test, y_test)

                                # FIX: limit CV folds to min(5, sample_size)
                                n_splits = min(5, clean.shape[0])
                                cv_scores = cross_val_score(model, X, y, cv=n_splits)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("R² Score (Test)", f"{score:.4f}")
                            col2.metric("CV Mean Score", f"{cv_scores.mean():.4f}")
                            col3.metric("CV Std Dev", f"{cv_scores.std():.4f}")

                            # Show CV scores chart
                            fig_cv = px.bar(
                                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                                y=cv_scores,
                                labels={"x": "Fold", "y": "R² Score"},
                                title="Cross-Validation Scores per Fold",
                            )
                            st.plotly_chart(fig_cv, use_container_width=True)

                            # Feature importance for RF
                            if model_choice == "Random Forest":
                                fi = pd.DataFrame({
                                    "Feature": feature_cols,
                                    "Importance": model.feature_importances_
                                }).sort_values("Importance", ascending=False)
                                fig_fi = px.bar(fi, x="Feature", y="Importance",
                                                title="Random Forest Feature Importances")
                                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.error(f"Model training error: {e}")