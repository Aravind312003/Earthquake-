import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Title
st.title("Earthquake Prediction using Deep Learning on Seismic Data")

# File upload
uploaded_file = st.file_uploader("Upload Seismic Data CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Data", df.head())

    # User selects target and features
    target_col = st.selectbox("Select Target Column (e.g., 'magnitude' or binary class)", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])

    if target_col and feature_cols:
        # Preprocessing
        X = df[feature_cols].values
        y = df[target_col].values

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Build model
        model = Sequential([
            Dense(64, input_dim=X.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear' if y.dtype != 'int' else 'sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error' if y.dtype != 'int' else 'binary_crossentropy',
            metrics=['mae'] if y.dtype != 'int' else ['accuracy']
        )

        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=0)
                st.success("Model trained successfully!")

        # Prediction
        st.write("## Predict on New Data")
        input_data = {}
        for feature in feature_cols:
            val = st.number_input(f"Enter value for {feature}")
            input_data[feature] = val

        if st.button("Predict"):
            input_array = np.array([list(input_data.values())])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            st.write("### Prediction Result:", prediction[0][0])




