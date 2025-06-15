import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# App title
st.title("🧠 Deep Learning App for Image or Data")

# Sidebar for upload
st.sidebar.header("📂 Upload Image or Dataset")

uploaded_image = st.sidebar.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png"])
uploaded_csv = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# ---------- IMAGE DISPLAY ----------
if uploaded_image is not None:
    st.subheader("🖼️ Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# ---------- CSV HANDLING ----------
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Data Summary")
    st.write(df.describe())

    # Drop all-null columns
    df.dropna(axis=1, how='all', inplace=True)

    # Label encode object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Select Features and Target
    st.subheader("🧪 Model Setup")
    features = st.multiselect("Select Input Features", df.columns.tolist())
    target = st.selectbox("Select Target Column (Classification)", df.columns.tolist())

    if features and target:
        df = df.dropna(subset=features + [target])
        df[features] = df[features].fillna(df[features].mean())

        X = df[features]
        y = LabelEncoder().fit_transform(df[target])

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # One-hot encode output if multi-class
        y_cat = to_categorical(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

        # Build model
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(y_cat.shape[1], activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        st.info("Training model, please wait...")
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=16, verbose=0)

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"✅ Test Accuracy: **{acc * 100:.2f}%**")

        # Plot training history
        st.subheader("📉 Training History")
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title("Accuracy over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        st.pyplot(fig1)

        # Confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        st.subheader("📊 Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        st.subheader("📝 Classification Report")
        st.text(classification_report(y_true, y_pred))

        # Prediction from custom input
        st.subheader("🔍 Predict with Custom Input")
        custom_input = []
        for f in features:
            val = st.number_input(f"Enter value for {f}", value=float(df[f].mean()))
            custom_input.append(val)

        if st.button("Predict Class"):
            input_array = scaler.transform([custom_input])
            prediction = np.argmax(model.predict(input_array), axis=1)[0]
            st.success(f"Predicted Class: {prediction}")

        # Optional visualization
        st.subheader("📈 Feature Visualization")
        plot_type = st.selectbox("Choose Plot", ["Histogram", "Correlation Heatmap"])
        if plot_type == "Histogram":
            col = st.selectbox("Select Column", features)
            fig3, ax3 = plt.subplots()
            ax3.hist(df[col], bins=20)
            ax3.set_title(f"Histogram of {col}")
            st.pyplot(fig3)
        else:
            fig4, ax4 = plt.subplots()
            sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", ax=ax4)
            st.pyplot(fig4)




