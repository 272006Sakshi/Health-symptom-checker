import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="AI Health Symptom Checker",
    page_icon="🩺",
    layout="wide"
)

# LOAD MODEL FILES

model = joblib.load("disease_model.pkl")
symptoms_list = joblib.load("symptoms.pkl")
encoder = joblib.load("label_encoder.pkl")

# SESSION STATE

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "top3_diseases" not in st.session_state:
    st.session_state.top3_diseases = []

if "top3_probs" not in st.session_state:
    st.session_state.top3_probs = []

# SIDEBAR

st.sidebar.title("🩺 Health Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "Disease Prediction",
        "Appointment History"
    ]
)

st.sidebar.info(
    """
    AI-powered disease prediction system.

    Select symptoms and get possible disease predictions.
    """
)

st.sidebar.success("System Status: Active ✅")

# DISEASE PREDICTION PAGE

if page == "Disease Prediction":

    st.title("🩺 AI Based Health Symptom Checker")

    st.markdown(
        """
        Welcome to the AI-powered healthcare assistant.

        Select your symptoms below to predict possible diseases.
        """
    )

    # SYMPTOM SELECTION

    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        symptoms_list
    )

    # PREDICT BUTTON

    if st.button("Predict Disease"):

        # Check symptom count
        if len(selected_symptoms) < 4:

            st.warning(
                "⚠ Please select at least 4 symptoms for better prediction accuracy."
            )

            st.session_state.prediction_done = False

        else:

            # Create input vector
            input_data = [0] * len(symptoms_list)

            for symptom in selected_symptoms:

                index = symptoms_list.index(symptom)

                input_data[index] = 1

            # Convert to numpy array
            input_data = np.array(input_data).reshape(1, -1)

            # MODEL PREDICTION

            probabilities = model.predict_proba(input_data)

            top3_indices = np.argsort(
                probabilities[0]
            )[-3:][::-1]

            top3_diseases = encoder.inverse_transform(
                top3_indices
            )

            top3_probs = probabilities[0][top3_indices]

            # Save in session state
            st.session_state.prediction_done = True

            st.session_state.top3_diseases = top3_diseases

            st.session_state.top3_probs = top3_probs

    # SHOW RESULTS AFTER PREDICTION

    if st.session_state.prediction_done:

        st.subheader("🩺 Prediction Result")

        st.success(
            f"Most Probable Disease: {st.session_state.top3_diseases[0]}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{st.session_state.top3_probs[0]*100:.2f}%"
        )
        
        # TOP 3 PREDICTIONS

        st.subheader("Top 3 Possible Diseases")

        for i in range(3):

            st.write(
                f"{i+1}. "
                f"{st.session_state.top3_diseases[i]} "
                f"→ "
                f"{st.session_state.top3_probs[i]*100:.2f}%"
            )

        # CONFIDENCE WARNING

        if st.session_state.top3_probs[0] < 0.30:

            st.warning(
                "⚠ Prediction confidence is low. "
                "Please provide more accurate symptoms."
            )

        # APPOINTMENT BOOKING

        st.subheader("📅 Book Appointment")

        # Doctor mapping
        doctor_mapping = {
            "Allergy": "Dermatologist",
            "GERD": "Gastroenterologist",
            "Heart attack": "Cardiologist",
            "Bronchial Asthma": "Pulmonologist"
        }

        suggested_doctor = doctor_mapping.get(
            st.session_state.top3_diseases[0],
            "General Physician"
        )

        # APPOINTMENT FORM

        with st.form("appointment_form"):

            patient_name = st.text_input(
                "Enter Patient Name"
            )

            age = st.number_input(
                "Enter Age",
                min_value=1,
                max_value=120,
                step=1
            )

            gender = st.selectbox(
                "Select Gender",
                ["Male", "Female", "Other"]
            )

            doctor = st.text_input(
                "Suggested Doctor",
                value=suggested_doctor
            )

            appointment_date = st.date_input(
                "Select Appointment Date"
            )

            submit_appointment = st.form_submit_button(
                "Book Appointment"
            )

            # SAVE TO DATABASE

            if submit_appointment:

                try:
                    conn = sqlite3.connect(
                        "database.db"
                    )

                    cursor = conn.cursor()

                    cursor.execute("""
                    INSERT INTO appointments (
                        patient_name,
                        age,
                        gender,
                        disease,
                        doctor,
                        appointment_date
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        patient_name,
                        age,
                        gender,
                        st.session_state.top3_diseases[0],
                        doctor,
                        str(appointment_date)
                    ))

                    conn.commit()

                    conn.close()

                    st.success(
                        "✅ Appointment Booked Successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Database Error: {e}"
                    )

# APPOINTMENT HISTORY PAGE

elif page == "Appointment History":

    st.title("📋 Appointment History")

    try:

        conn = sqlite3.connect("database.db")

        cursor = conn.cursor()

        cursor.execute("""
        SELECT
            patient_name,
            age,
            gender,
            disease,
            doctor,
            appointment_date
        FROM appointments
        """)

        data = cursor.fetchall()

        conn.close()

        if len(data) == 0:

            st.warning(
                "No appointments booked yet."
            )

        else:

            df = pd.DataFrame(
                data,
                columns=[
                    "Patient Name",
                    "Age",
                    "Gender",
                    "Disease",
                    "Doctor",
                    "Appointment Date"
                ]
            )

            st.dataframe(
                df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Database Error: {e}"
        )

# DISCLAIMER

st.markdown("---")

st.caption(
    "⚠ This system provides AI-based predictions "
    "and should not replace professional medical consultation."
)