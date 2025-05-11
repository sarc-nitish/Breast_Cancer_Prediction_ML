# Breast_Cancer_Prediction_ML

Breast Cancer Prediction Using Neural Networks
Description
This project develops a machine learning model to classify breast tumors as Malignant (M) or Benign (B) using a neural network built with TensorFlow/Keras. The model analyzes key tumor features to predict cancer diagnosis with high accuracy. It includes data preprocessing, feature selection, model training, evaluation, and a Streamlit web app for real-time predictions, deployed on Streamlit Cloud.
[Live Demo]: Try the prediction app: Breast Cancer Prediction App
Features
click here--> https://sarcbreastcancerprediction.streamlit.app

Classifies breast tumors as Malignant or Benign.
Achieves ~89.47% accuracy on the test dataset.
Real-time predictions via a user-friendly Streamlit web app.
Data preprocessing with feature selection and standardization.
Visualizes training performance with loss curves.

Tech Stack

Programming Language: Python
Libraries/Frameworks:
TensorFlow/Keras (neural network)
Pandas (data manipulation)
NumPy (numerical operations)
Scikit-learn (preprocessing)
Matplotlib (visualization)
Streamlit (web app)


Tools: Jupyter Notebook, Git, Streamlit Cloud

Dataset

Source: UCI Breast Cancer Wisconsin (Diagnostic) (included as Breast_Cancer_data.csv).
Description: 569 samples, 32 features (e.g., radius, perimeter, concavity). Target: diagnosis (M = Malignant, B = Benign).
Preprocessing:
Dropped id and Unnamed: 32 columns.
Selected 10 features: radius_mean, perimeter_mean, area_mean, concavity_mean, concave points_mean, radius_worst, perimeter_worst, area_worst, concavity_worst, concave points_worst.
Standardized using StandardScaler.



Installation

Clone the repository:git clone <your-repository-link>


Navigate to the project directory:cd <repository-folder>


Install dependencies:pip install -r requirements.txt



Usage
Local Execution

Open the Jupyter Notebook:jupyter notebook Breast_cancer_prediction.ipynb


Run cells to preprocess data, train, and evaluate the model.
Run the Streamlit app locally:streamlit run app.py



Real-Time Predictions

Streamlit App: Visit Breast Cancer Prediction App and input tumor features.

Notebook: Load Breast_Cancer_Model.keras and scaler.pkl, input 10 features, and predict.
 Example input:
input_data = (7.76, 47.92, 181.0, 0.0, 0.0, 9.456, 59.16, 268.6, 0.0, 0.0)



Results

Accuracy: ~89.47% on test data.
Loss Curves: Training/validation loss plotted (see loss_plot.png, if saved).
Example Prediction:
Input: (7.76, 47.92, 181.0, 0.0, 0.0, 9.456, 59.16, 268.6, 0.0, 0.0)
Output: Benign (~99.69% probability).
click here --> https://sarcbreastcancerprediction.streamlit.app



Future Work

Compare with other models (e.g., SVM, Random Forest).
Add hyperparameter tuning for better accuracy.
Enhance Streamlit app with prediction confidence and visualizations.
Include confusion matrix and ROC curve.

File Structure

Breast_cancer_prediction.ipynb: Model training and evaluation.
app.py: Streamlit web app script.
Breast_Cancer_data.csv: Original dataset.
Cleaned_Brest_Cancer_data.csv: Preprocessed dataset.
Breast_Cancer_Model.keras: Saved model.
scaler.pkl: Saved scaler.
requirements.txt: Dependencies.

Contact

LinkedIn: https://www.linkedin.com/in/nitish-kumar-sarc
Email: nitish75577@gmail.com


Built to demonstrate skills in machine learning, neural networks, and web deployment for internship applications.
