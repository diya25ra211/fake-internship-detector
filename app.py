
import streamlit as st
import joblib
import pandas as pd

# Load the trained CatBoost model
model = joblib.load("catboost_model.pkl")

st.title("🚨 Fake Internship Postings Detector")
st.write(
    "Enter the internship details below to check if "
    "it's legitimate or a potential scam."
)

# Input widgets
col1, col2 = st.columns(2)

with col1:
    urgency_score = st.slider(
        "Urgency Score (0-100)", 0.0, 100.0, 25.0
    )
    vague_description_score = st.slider(
        "Vague Description Score (0-100)", 0.0, 100.0, 20.0
    )
    phishing_language_score = st.slider(
        "Phishing Language Score (0-100)", 0.0, 100.0, 10.0
    )
    trust_signal_score = st.slider(
        "Trust Signal Score (0-100)", 0.0, 100.0, 50.0
    )
    keyword_spam_score = st.slider(
        "Keyword Spam Score (0-100)", 0.0, 100.0, 15.0
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-Time", "Internship", "Part-Time"]
    )

    work_mode = st.selectbox(
        "Work Mode",
        ["Onsite", "Remote"]
    )

    industry = st.selectbox(
        "Industry",
        [
            "Cybersecurity",
            "E-Commerce",
            "EdTech",
            "FinTech",
            "Gaming",
            "Healthcare",
            "Marketing",
            "Software"
        ]
    )

with col2:
    job_description_length = st.number_input(
        "Job Description Length (characters)",
        min_value=100, max_value=5000, value=1500
    )

    stipend = st.number_input(
        "Stipend Amount (₹)",
        min_value=0, max_value=200000, value=25000
    )

    recruiter_response_time = st.number_input(
        "Recruiter Response Time (hours)",
        min_value=1.0, max_value=72.0, value=12.0
    )

    domain_age_months = st.number_input(
        "Domain Age (months)",
        min_value=1, max_value=600, value=120
    )

    recruiter_experience_years = st.number_input(
        "Recruiter Experience (years)",
        min_value=0.0, max_value=20.0, value=4.0
    )

    location = st.selectbox(
        "Company Location",
        [
            "Berlin",
            "Dubai",
            "London",
            "New York",
            "San Francisco",
            "Singapore",
            "Sydney",
            "Toronto"
        ]
    )

    company_size = st.selectbox(
        "Company Size",
        ["Medium", "Small", "Startup"]
    )


if st.button("Check Internship"):

    # Default numeric and categorical features
    data = {
        "company_age": 15.0,
        "linkedin_presence": 1.0,
        "website_available": 1.0,
        "domain_age_months": domain_age_months,
        "verification_status": 1.0,
        "stipend": stipend,
        "payment_required": 0.0,
        "registration_fee": 0.0,
        "job_description_length": job_description_length,
        "grammatical_errors": 1.0,
        "vague_description_score": vague_description_score,
        "urgency_score": urgency_score,
        "keyword_spam_score": keyword_spam_score,
        "fake_certificate_offer": 0.0,
        "recruiter_experience_years": recruiter_experience_years,
        "suspicious_email_domain": 0.0,
        "recruiter_email_type_Free": 0.0,
        "recruiter_response_time_hours": recruiter_response_time,
        "social_media_presence": 1.0,
        "emotional_manipulation_score": 20.0,
        "phishing_language_score": phishing_language_score,
        "trust_signal_score": trust_signal_score,
        "posting_year": 2026,
        "posting_month": 6,
        "posting_day": 1,
        "posting_dayofweek": 2,
        "phishing_risk_tier": 2,
        "urgency_risk_tier": 2,
        "vague_risk_tier": 2,
        "total_suspicious_flags": 1,
    }

    # Employment type one-hot encoding
    for value in ["Full-Time", "Internship", "Part-Time"]:
        data[f"employment_type_{value}"] = int(
            employment_type == value
        )

    # Work mode one-hot encoding
    for value in ["Onsite", "Remote"]:
        data[f"work_mode_{value}"] = int(work_mode == value)

    # Industry one-hot encoding
    industries = [
        "Cybersecurity", "E-Commerce", "EdTech", "FinTech",
        "Gaming", "Healthcare", "Marketing", "Software"
    ]

    for value in industries:
        data[f"industry_{value}"] = int(industry == value)

    # Location one-hot encoding
    locations = [
        "Berlin", "Dubai", "London", "New York",
        "San Francisco", "Singapore", "Sydney", "Toronto"
    ]

    for value in locations:
        data[f"location_{value}"] = int(location == value)

    # Company size one-hot encoding
    company_sizes = ["Medium", "Small", "Startup"]

    for value in company_sizes:
        data[f"company_size_{value}"] = int(
            company_size == value
        )

    # Create the input DataFrame
    input_data = pd.DataFrame([data])

    try:
        # Get the exact feature names and order from the model
        expected_features = model.feature_names_

        # Check for missing features
        missing_features = [
            feature for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        else:
            # Match the training feature order
            input_data = input_data[expected_features]

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                st.error(
                    "Class 1 is not present in the model. "
                    "Check your training labels."
                )
            else:
                fake_index = classes.index(1)
                probability = probabilities[fake_index] * 100

                if prediction == 1:
                    st.error(
                        "⚠️ Warning: This posting is likely FAKE!"
                    )
                    st.write(
                        f"Estimated scam probability: "
                        f"{probability:.2f}%"
                    )
                else:
                    st.success(
                        "✅ This posting appears to be Legitimate."
                    )
                    st.write(
                        f"Estimated real probability: "
                        f"{100 - probability:.2f}%"
                    )

    except Exception as e:
        st.error(f"Prediction error: {e}")
