import streamlit as st # Import the Streamlit library for the web interface
import pandas as pd # Import Pandas for data handling
from sklearn.ensemble import RandomForestClassifier # Import the ML algorithm
from sklearn.model_selection import train_test_split # Import data splitting tool
from sklearn.preprocessing import LabelEncoder # Import tool to convert text to numbers

# Set up the web page tab title and layout
st.set_page_config(page_title="Student Placement Predictor", layout="centered")

# Use st.cache_resource so the model only trains ONCE when the app starts, not every time a user clicks a button
@st.cache_resource 
def train_model():
    # 1. Load the dataset
    try:
        data = pd.read_csv("student_placement/placement-data.csv")
    except FileNotFoundError:
        return None, None, None, "Dataset not found. Please ensure 'placement-data.csv' is uploaded."

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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_status, features, "Success"

# Run the training function
model, le_status, features, status_message = train_model()

# --- WEB INTERFACE DESIGN ---

st.title("🎓 Student Placement Predictor")
st.write("Enter the student's details below to predict if they will be placed during campus recruitment.")

if model is None:
    st.error(status_message) # Show an error if the CSV is missing
else:
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
        
        # Make the prediction
        prediction_num = model.predict(input_data)
        
        # Convert the predicted number (1 or 0) back to text ("Placed" or "NotPlaced")
        predicted_text = le_status.inverse_transform(prediction_num)[0]
        
        st.divider() # Draw a line on the screen
        
        # Display results with colors
        if predicted_text == "Placed":
            st.success("🎉 Prediction: **This student is highly likely to be PLACED.**")
        else:
            st.error("⚠️ Prediction: **This student is likely NOT PLACED.**")
