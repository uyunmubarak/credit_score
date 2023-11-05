#import library
import streamlit as st
import requests
import pandas as pd
import json

# Set the background color using CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


#set the title of the Streamlit app
st.title("Prediction Credit Score")
st.markdown("_Please fill in the applicant data in the form below._")

#create a form to collect applicant's data
with st.form(key = "applicant_data_form"):

    #input fields for the applicant's data
    app_name = st.text_input('**Name :**', '')

    age=st.number_input(
        label = "1.\t**Age :**",
        min_value = 20,
        max_value = 60,
        help = "Value range from 20 to 60"
    )

    income = st.number_input(
        label = "2.\t**Total income (USD) :**",
        min_value = 15000,
        max_value = 149999,
        help = "Value range from 15000 to 149999"
    )

    loan_amount = st.number_input(
        label = "3.\t**Total loan amount (USD) :**",
        min_value = 5000,
        max_value = 249999,
        help = "Value range from 5000 to 249999"
    )

    months_employed = st.number_input(
        label = "4.\t**Total months employed :**",
        min_value = 0,
        max_value = 119,
        help = "Value range from 0 to 119"
    )

    num_credit_lines = st.number_input(
        label = "5.\t**Total number credit lines :**",
        min_value = 1,
        max_value = 4,
        help = "Value range from 1 to 4"
    )

    interest_rate = st.number_input(
        label = "6.\t**Interest rate :**",
        min_value = 2,
        max_value = 25,
        help = "Value range from 2 to 25"
    )

    loan_term = st.number_input(
        label = "7.\t**Loan term :**",
        min_value = 12,
        max_value = 60,
        help = "Value range from 12 to 60"
    )

    DTI_ratio = st.number_input(
        label = "8.\t**DTI ratio :**",
        min_value = 0.10,
        max_value = 0.90,
        help = "Value range from 0.10 to 0.90"
    )

    education = st.radio(
        label = "9.\t**Education :**",
        options = ("High School", "Bachelor's" ,"Master's", "PhD"),
        index = 0,
        horizontal = True
    )

    employment_type = st.radio(
        label = "10.\t**Employment type :**",
        options = ("Part-time", "Full-time", "Self-employed", "Unemployed")
    )

    marital_status = st.radio(
        label = "11.\t**Marital status :**",
        options = ("Single", "Married", "Divorced"),
        index = 0,
        horizontal = True
    )

    has_mortgage = st.radio(
        label = "12.\t**Has mortgage? :**",
        options = ("Yes", "No"),
        index = 0,
        horizontal = True
    )

    has_dependents = st.radio(
        label = "13.\t**Has dependents? :**",
        options = ("Yes", "No"),
        index = 0,
        horizontal = True
    )

    loan_purpose = st.radio(
        label = "14.\t**Loan purpose :**",
        options = ("Business", "Home", "Education", "Other", "Auto")
    )

    has_co_signer = st.radio(
        label = "15.\t**Has co signer? :**",
        options = ("Yes", "No"),
        index = 0,
        horizontal = True
    )

    #submit button to trigger the prediction
    submitted = st.form_submit_button("**PREDICT**")

    if submitted:
        #cCollect the data from the input fields
        applicant_data_form = {
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": DTI_ratio,
            "Education": education,
            "EmploymentType": employment_type,
            "MaritalStatus": marital_status,
            "HasMortgage": has_mortgage,
            "HasDependents": has_dependents,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": has_co_signer
        }

        try:
            with st.spinner("Send data for predict server ..."):
                #send a POST request to the prediction server
                res = requests.post("http://127.0.0.1:8000/predict", json=applicant_data_form).json()
        except json.decoder.JSONDecodeError:
            st.error("Failed to decode JSON response from the server. The server response may not be valid JSON.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the server: {e}")
        else:
            #display the prediction results
            st.write(res)

            if 'Score' in res:
                #display the credit score, probability, and recommendation
                st.success(f"""
                    Applicant's name: **{app_name}**
                    
                    Credit score: **{res['Score']}**  
                    Probability of being good: **{res['Proba']}**  
                    Recommendation: **{res['Recommendation']}**
                """)
            else:
                st.error("Response does not contain 'Score' key.")
