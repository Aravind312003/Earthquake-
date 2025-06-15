import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌍 Earthquake Classification App (ML & Deep Learning)")

# Sidebar
st.sidebar.header("📤 Upload Data or Image")
uploaded_csv = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
uploaded_image = st.sidebar.file_uploader("Upload Image (optional)", type=["png", "jpg", "jpeg"])

# Show uploaded image
if uploaded_image:
    st.subheader("🖼️ Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Handle uploaded CSV
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Clean and encode
    df.dropna(axis=1, how='all', inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    st.subheader("🔍 Data Summary")
    st.write(df.describe())

    st.subheader("⚙️ Model Setup")
    features = st.multiselect("Select Feature Columns", df.columns.tolist())
    target = st.selectbox("Select Target Column", df.columns.tolist())

    if features and target and target not in features:
        df.dropna(subset=features + [target], inplace=True)
        df[features] = df[features].fillna(df[features].mean())

        X = df[features]
        y = LabelEncoder().fit_transform(df[target])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Test Accuracy: {acc * 100:.2f}%")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔢 Confusion Matrix")
        fig1, ax1 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax1)
        st.pyplot(fig1)

        # Prediction from custom input
        st.subheader("🧠 Predict Earthquake Class")
        custom_input = []
        for f in features:
            val = st.number_input(f"Enter value for {f}", value=float(df[f].mean()))
            custom_input.append(val)

        if st.button("📌 Predict Class"):
            input_scaled = scaler.transform([custom_input])
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Class: {prediction}")

        # Expanded Visualization Section
        st.subheader("📊 Feature Visualization")
        viz_type = st.selectbox("Choose Visualization Type", ["Histogram", "Boxplot", "Violinplot", "KDE", "Pairplot", "Correlation Heatmap"])

        if viz_type != "Correlation Heatmap" and viz_type != "Pairplot":
            for col in features:
                st.markdown(f"### {viz_type} of `{col}`")
                fig, ax = plt.subplots()
                if viz_type == "Histogram":
                    ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
                    ax.set_ylabel("Frequency")
                elif viz_type == "Boxplot":
                    sns.boxplot(x=df[col], ax=ax, color='lightgreen')
                elif viz_type == "Violinplot":
                    sns.violinplot(x=df[col], ax=ax, color='lightcoral')
                elif viz_type == "KDE":
                    sns.kdeplot(df[col], ax=ax, fill=True, color='purple')
                ax.set_xlabel(col)
                ax.set_title(f"{viz_type} for {col}")
                st.pyplot(fig)

        elif viz_type == "Correlation Heatmap":
            st.markdown("### 🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_matrix = df[features + [target]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif viz_type == "Pairplot":
            st.markdown("### 🤝 Pairplot")
            sampled_df = df[features + [target]].sample(n=min(200, len(df)))  # Limit to 200 rows to avoid overload
            fig = sns.pairplot(sampled_df, hue=target, palette='Set2')
            st.pyplot(fig)











