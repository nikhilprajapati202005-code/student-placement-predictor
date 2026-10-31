import streamlit as st # Import the Streamlit library for the web interface
import pandas as pd # Import Pandas for data handling
import joblib # Import joblib to save/load the trained model (no retraining on restart)
import os # Import os to check if a saved model file exists
from pathlib import Path # Import Path to locate files next to this script (works on any computer)
from sklearn.ensemble import RandomForestClassifier # Import the ML algorithm
from sklearn.model_selection import train_test_split # Import data splitting tool
from sklearn.preprocessing import LabelEncoder # Import tool to convert text to numbers

# Set up the web page tab title and layout
st.set_page_config(page_title="Student Placement Predictor", layout="centered")

# Always find data and model files in the same folder as app.py.
# Streamlit Cloud starts from the repo root, so naive relative paths fail there.
BASE_DIR = Path(__file__).parent

# Use st.cache_resource so the model only trains ONCE when the app starts, not every time a user clicks a button
@st.cache_resource 
def train_model():
    # 0. If we trained before, load the saved model instead (much faster startup)
    model_path = BASE_DIR / "placement_model.pkl"
    meta_path = BASE_DIR / "placement_meta.pkl"
    if model_path.exists() and meta_path.exists():
        try:
            model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            return model, meta["le_status"], meta["features"], meta["accuracy"], meta["n_students"], "Model loaded from saved file"
        except Exception:
            pass # If loading fails for any reason, just retrain below

    # 1. Load the dataset
    try:
        data = pd.read_csv(BASE_DIR / "placement-data.csv") 
    except FileNotFoundError:
        return None, None, None, None, None, "Dataset not found. Please ensure 'placement-data.csv' is uploaded."

    # 2. Preprocess the data (convert Yes/No to 1/0)
    le_extra = LabelEncoder()
    data['ExtracurricularActivities'] = le_extra.fit_transform(data['ExtracurricularActivities'])
    
    le_training = LabelEncoder()
    data['PlacementTraining'] = le_training.fit_transform(data['PlacementTraining'])
    
    le_status = LabelEncoder()
    data['PlacementStatus'] = le_status.fit_transform(data['PlacementStatus'])

    # 3. Define features (inputs) and target (output)
    features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities', 
                'PlacementTraining', 'SSC_Marks', 'HSC_Marks']
    X = data[features]
    y = data['PlacementStatus']

    # 4. Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # max_depth and min_samples_leaf stop the trees from memorizing the training data (overfitting)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Measure how accurate the model is on test data it has NEVER seen before
    accuracy = model.score(X_test, y_test) # e.g. 0.83 means 83% of predictions were correct
    
    # 6. Save the trained model to disk so the next app start doesn't need to retrain.
    # On some hosting services the folder is read-only, so saving is optional — the app
    # simply retrains on the next start if saving isn't possible.
    try:
        joblib.dump(model, model_path)
        joblib.dump(
            {"le_status": le_status, "features": features, "accuracy": accuracy, "n_students": len(data)},
            meta_path
        )
    except Exception:
        pass # Saving failed (e.g. read-only filesystem); the model still works for this session
    
    return model, le_status, features, accuracy, len(data), "Trained a new model"

# Run the training function
model, le_status, features, accuracy, n_students, status_message = train_model()

# --- WEB INTERFACE DESIGN ---

st.title("🎓 Student Placement Predictor")
st.write("Enter the student's details below to predict if they will be placed during campus recruitment.")

if model is None:
    st.error(status_message) # Show an error if the CSV is missing
else:
    # Show the model's real performance in the sidebar
    with st.sidebar:
        st.header("📊 About the Model")
        st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")
        st.metric("Students in Dataset", f"{n_students:,}")
        st.caption(f"Source: {status_message} • 80% of data used for training, 20% held out for testing")
    # Create two columns to make the form look neat
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.number_input("CGPA (0.0 to 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        interns = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
        projs = st.number_input("Number of Projects", min_value=0, max_value=20, value=2)
        workshops = st.number_input("Workshops/Certifications", min_value=0, max_value=20, value=1)
        aptitude = st.number_input("Aptitude Test Score (0-100)", min_value=0, max_value=100, value=75)
        
    with col2:
        soft_skills = st.number_input("Soft Skills Rating (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        ssc = st.number_input("SSC Marks % (10th Grade)", min_value=0, max_value=100, value=75)
        hsc = st.number_input("HSC Marks % (12th Grade)", min_value=0, max_value=100, value=75)
        extra_input = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        training_input = st.selectbox("Placement Training Done?", ["Yes", "No"])

    # Predict Button
    if st.button("Predict Placement", type="primary"):
        # Convert dropdown "Yes"/"No" back into 1/0 for the model
        extra_val = 1 if extra_input == "Yes" else 0
        training_val = 1 if training_input == "Yes" else 0
        
        # Package user inputs into a DataFrame
        input_data = pd.DataFrame([[cgpa, interns, projs, workshops, aptitude, soft_skills, extra_val, training_val, ssc, hsc]], 
                                  columns=features)
        
        # Make the prediction AND get the model's confidence level
        prediction_num = model.predict(input_data)
        pred_probs = model.predict_proba(input_data)[0] # e.g. [0.88, 0.12]
        
        # Convert the predicted number (1 or 0) back to text ("Placed" or "NotPlaced")
        predicted_text = le_status.inverse_transform(prediction_num)[0]
        
        # Pick the probability that matches the predicted class
        confidence = pred_probs[list(le_status.classes_).index(predicted_text)] * 100
        
        st.divider() # Draw a line on the screen
        
        # Display results with colors, now including how confident the model is
        if predicted_text == "Placed":
            st.success(f"🎉 Prediction: **This student is highly likely to be PLACED.** (Model confidence: {confidence:.1f}%)")
        else:
            st.error(f"⚠️ Prediction: **This student is likely NOT PLACED.** (Model confidence: {confidence:.1f}%)")
        
        st.progress(confidence / 100, text=f"Model confidence: {confidence:.1f}%")
        
        # Feature importance chart: show WHICH features the Random Forest relied on most
        with st.expander("🔍 Which features influenced the prediction the most?"):
            st.write("The Random Forest automatically learns how important each feature is when deciding:")
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(importances)