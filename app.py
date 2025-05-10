# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib

# # Load trained model and scaler
# model = tf.keras.models.load_model("Breast_Cancer_Prediction.h5")
# scaler = joblib.load("scaler.pkl")

# st.title("🔬 Breast Cancer Prediction App")
# st.markdown("Enter the following details to predict whether the tumor is **Benign** or **Malignant**.")

# # ✅ Correct feature order inputs
# radius_mean = st.number_input("Radius Mean")
# perimeter_mean = st.number_input("Perimeter Mean")
# area_mean = st.number_input("Area Mean")
# concavity_mean = st.number_input("Concavity Mean")
# concave_points_mean = st.number_input("Concave Points Mean")
# radius_worst = st.number_input("Radius Worst")
# perimeter_worst = st.number_input("Perimeter Worst")
# area_worst = st.number_input("Area Worst")
# concavity_worst = st.number_input("Concavity Worst")
# concave_points_worst = st.number_input("Concave Points Worst")

# if st.button("Predict"):
#     # Prepare input
#     input_data = np.array([[radius_mean, perimeter_mean, area_mean,
#                             concavity_mean, concave_points_mean,
#                             radius_worst, perimeter_worst, area_worst,
#                             concavity_worst, concave_points_worst]])

#     # Scale input
#     input_scaled = scaler.transform(input_data)

#     # Predict
#     prediction = model.predict(input_scaled)  # shape: (1, 2)
#     predicted_class = np.argmax(prediction, axis=1)[0]

#     # Show outputs
#     st.write("🔍 Raw prediction probabilities:", prediction)
#     st.write("🔢 Predicted class:", predicted_class)

#     # Final result
#     st.subheader("🩺 Prediction Result:")
#     if predicted_class == 1:
#         st.error("The model predicts: ** Benign Tumor** ")
#     else:
#         st.success("The model predicts: ** Malignant Tumor** ")




# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib

# # Load model and scaler
# model = tf.keras.models.load_model("Breast_Cancer_Prediction.h5")
# scaler = joblib.load("scaler.pkl")

# st.title("🔬 Breast Cancer Prediction App")
# st.markdown("Enter the 10 features below (comma or space/tab separated):")
# st.markdown("**Order:** radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, radius_worst, perimeter_worst, area_worst, concavity_worst, concave_points_worst")

# # Text input box
# input_str = st.text_input("Paste your input (10 values)", "20.60 140.10 1265.0 0.35140 0.15200 25.740 184.60 1821.0 0.9387 0.26500")

# if st.button("Predict"):
#     try:
#         # Parse input
#         parts = input_str.replace(",", " ").split()
#         if len(parts) != 10:
#             st.error("⚠️ Please enter exactly 10 numeric values.")
#         else:
#             input_data = np.array([float(i) for i in parts]).reshape(1, -1)

#             # Scale input and predict
#             input_scaled = scaler.transform(input_data)
#             prediction = model.predict(input_scaled)
#             predicted_class = np.argmax(prediction, axis=1)[0]

#             # Output
#             st.subheader("🩺 Prediction Result:")
#             st.write("🔍 Model output (probabilities):", prediction)
#             st.write("🔢 Predicted class:", predicted_class)

#             if predicted_class == 1:
#                 st.error("The model predicts: **Malignant Tumor** 💥")
#             else:
#                 st.success("The model predicts: **Benign Tumor** ✅")
#     except Exception as e:
#         st.error(f"⚠️ Error in input: {e}")



import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model("Breast_Cancer_Model.keras")  # Updated to use .keras format
scaler = joblib.load("scaler.pkl")

st.write("Hi I'm Nitish.  This My 1st Neural Network Project")
st.title("Breast Cancer Prediction ")
st.markdown("Enter the following details to predict whether the tumor is **Benign** or **Malignant**.")

# Input for 10 features in a single input
st.write("Radius Mean  , Perimeter Mean ,Area Mean , Concavity Mean , Concave Points Mean , Radius Worst ,Perimeter Worst , Area Worst , Concavity Worst , Concave Points Worst ")
input_data = st.text_input("Enter 10 features separated by commas in Order (like this--> 20.60, 140.10):")



# Prediction
if st.button("Predict") and input_data:
    try:
        # Convert input data to numpy array
        input_data_list = [float(i) for i in input_data.split(",")]
        input_data_array = np.array([input_data_list])

        # Standardize the input data
        input_scaled = scaler.transform(input_data_array)

        # Make prediction
        prediction = model.predict(input_scaled)  # shape will be (1, 2)

        # Get predicted class (0: Benign, 1: Malignant)
        predicted_class = np.argmax(prediction, axis=1)[0]

        #st.write("🔢 Predicted class  (0: Benign, 1: Malignant):", predicted_class)
        #st.write("🔍 Raw prediction:", prediction)

        # Show result
        st.subheader("Prediction Result:")
        if predicted_class == 1:
            st.error("The model predicts: Benign Tumor✅")
        else:
            st.success("The model predicts: Malignant Tumor......!!!")
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")
else:
    st.info("Please enter the 10 features to make a prediction.")
