import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


thyroid = pd.read_csv('data/thyroid_cancer_risk_data.csv')
thyroid = thyroid.drop(columns= 'Patient_ID')

# Page Title and Icon
st.set_page_config(page_title= "Thyroid Cancer Predictions", page_icon= "🦠")

# Sidebar Navigation
page = st.sidebar.selectbox("Select a Page",["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions"])

# Home page
if page == "Home":
    st.title("🦠 Thyroid Cancer Analysis")
    st.subheader("Welcome to our Thyroid Cancer Explorer App")

    st.divider()

    st.write("""This analysis simulates real-world thyroid cancer risk factors using a dataset from 
             [Kaggle](https://www.kaggle.com/datasets/ankushpanday1/thyroid-cancer-risk-prediction-dataset/data),
             providing insights into key contributors to the disease and identifying regions where thyroid cancer is most prevalent.
             The goal is to determine what makes a patient high-risk and which personal health factors they should monitor to stay proactive about potential thyroid cancer concerns.
             """)
    
    st.divider()

    st.image('https://www.ent-newyork.com/images/anaplastic-thyroid-lymphoma-img1.jpg')
    st.markdown("Image used from [ent-newyork.com](https://www.ent-newyork.com/thyroid-lymphoma-nyc.htm)")


# Data Overview
elif page == "Data Overview":
    st.title("📋 Data Overview")
    st.subheader("About the Data")
    st.write("""The thyroid is a vital gland in the body, responsible for producing hormones that regulate metabolism, heart rate, breathing, and bone growth.
             In this dataset, we analyze 212,691 patients based on various factors such as gender, country of origin, ethnicity, family history, and iodine exposure, among others.
             Through this analysis, we aim to assess patients' risk factors and determine their likelihood of developing thyroid cancer.
             """)
    
    st.divider()

    st.image("https://media.istockphoto.com/id/108272967/photo/surgeon.jpg?s=612x612&w=0&k=20&c=wDZU_9nfxpisjLv5WDsohDXl_AkJt95dxM_xRH9YIzA=")
    st.markdown("Image used from [istockphoto.com](https://www.istockphoto.com/photo/surgeon-gm108272967-10544552)")


    st.divider()

    # Shape of data
    st.subheader("Quick Glance of Data")
    st.markdown(f"""The Thyroid Cancer Dataset consists of {thyroid.shape[0]} rows and {thyroid.shape[1]} columns and 
                is sourced from [Kaggle](https://www.kaggle.com/datasets/ankushpanday1/thyroid-cancer-risk-prediction-dataset/data).  
                Click one below to view.""")
    
     # Dataframe
    if st.checkbox("Show DataFrame"):
        st.dataframe(thyroid)
    # Dictionary
    if st.checkbox("Show Dictionary"):
        st.markdown(
        """
        | Column Name | Description |
        |-------------|-------------|
        | Patient_ID     | Observation number |
        | Age     | Age of patient |
        | Gender     | Gender of patient (Female or Male) |
        | Country     |  Identifies the country of origin of the patient |
        | Ethnicity     | Identifies the ethnic background |
        | Family History    | Identifies Yes or No for a family history of thyroid cancer |
        | Radiation Exposure     | Identifies Yes or No if the patient has been exposed to radiation |
        | Iodine Deficiency    | Identifies Yes or No if the patient is deficient in iodine levels |
        | Smoking    | Identifies Yes or No if the patient smokes |
        | Obesity    | Identifies Yes or No if the patient is obesee|
        | Diabetes     | Identifies Yes or No if the patient is diabetic |
        | TSH levels    | Provides the value level of Thyroid-Stimulating Hormone (TSH) |
        | T3 Levels    | Provides the value level of Triiodothyronine (T3) |
        | T4 Levels     |  Provides the value level of Thyroxine (T4) |
        | Nodule Size     | Provides the size of the nodule |
        | Thyroid Cancer Risk     | Identifies the patient's risk of thyroid cancer as Low, Medium, or High |
        | Diagnosis     | Identifies whether the patient has thyroid cancer (Malignant or Benign) |
        """,
        unsafe_allow_html=True)


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Key visualization to explore.")

    st.divider()

    st.markdown("#### Visualization 1: Country Diagnosis Comparison")
    st.markdown("This shows that India has a 12% higher rate of thyroid cancer compared to other countries.")
    st.image("data_images/visualization_1.PNG", caption="Visualization 1")

    st.divider()

    st.markdown("#### Visualization 2: Ethnicity Diagnosis Comparisons")
    st.markdown("This shows that Asians have a higher percentage of thyroid cancer cases—about 8% more than the next highest ethnicity, Africans, who have a rate of 25%.")
    st.image("data_images/visualization_2.PNG", caption="Visualization 2")

    st.divider()

    st.markdown("#### Visualization 3: Family History Diagnosis Comparison")
    st.markdown("Having a family history of thyroid cancer increases the risk by 13%, compared to 19% for those with no family histor.")
    st.image("data_images/visualization_3.PNG", caption="Visualization 3")

    st.divider()

    st.markdown("#### Visualization 4: Radiation Exposure Diagnosis Comparison")
    st.markdown("The data shows that exposure to radiation increases the risk of thyroid cancer by 11% compared to those with no exposure.")
    st.image("data_images/visualization_4.PNG", caption="Visualization 4")

    st.divider()

    st.markdown("#### Visualization 5: Iodine Deficiency Diagnosis Comparison")
    st.markdown("Having low iodine levels increases the risk of thyroid cancer by 10%, compared to an efficiency level of 21% in individuals with sufficient iodine.")
    st.image("data_images/visualization_5.PNG", caption="Visualization 5")

    st.divider()

    st.markdown("#### Visualization 6: Correlations with Diagnosis")
    st.markdown("We can now observe both positive and negative correlations. Thyroid cancer risk has the highest positive correlation at 0.37, followed by family history (0.14), Asian ethnicity (0.14), and Indian origin (0.11).")
    st.image("data_images/visualization_6.PNG", caption="Visualization 6")

    st.divider()

    st.markdown("#### Visualization 7: KNeighbors Classifier Confusion Matrix")
    st.markdown("With n_neighbors set to 51, the model's performance improved, increasing True Negative predictions by 719 and True Positive predictions by 610.")
    st.image("data_images/visualization_7.PNG", caption="Visualization 7")


# Model Training and Evaluation
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    train_df = pd.read_csv('data/cleaned_thyroid_data.csv')
    train_df = train_df.drop(columns =['Unnamed: 0', 'patient_id'])

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    features = train_df.drop(columns = 'diagnosis')
    X = features
    y = train_df['diagnosis']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    # StandardScaler
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.number_input("Select the number of neighbors (k) from 1-51", min_value=1, max_value=51, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()
    
    # Train the model on the scaled data
    model.fit(X_train_sc, y_train)

     # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_sc, y_train):.6f}")
    st.write(f"Test Accuracy: {model.score(X_test_sc, y_test):.6f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_sc, y_test, ax=ax, cmap='Greys')
    st.pyplot(fig)
        
    st.divider()


# Make Predictions Page
elif page == "Make Predictions":
    st.title("✈️ Make Predictions")
    st.subheader("Adjust the values below to make predictions on the Thyroid dataset:")

    train_df = pd.read_csv('data/cleaned_thyroid_data.csv')
    train_df = train_df.drop(columns =['Unnamed: 0', 'patient_id'])

   # Separate features (X) and target (y)
    X = train_df.drop(columns=["diagnosis"])
    y = train_df["diagnosis"]

    # Extract country and ethnicity columns
    country_columns = [col for col in X.columns if col.startswith("country_")]
    ethnicity_columns = [col for col in X.columns if col.startswith("ethnicity_")]

    # Get country and ethnicity names by removing prefixes
    countries = [col.replace("country_", "") for col in country_columns]
    ethnicities = [col.replace("ethnicity_", "") for col in ethnicity_columns]

    # Define categorical options
    gender = ["Male", "Female"]
    history = ["No", "Yes"]
    exposure = ["No", "Yes"]
    deficiency = ["No", "Yes"]
    smoking = ["No", "Yes"]
    obesity = ["No", "Yes"]
    diabetes = ["No", "Yes"]
    risk = ["Low", "Medium", "High"]

    # Streamlit UI
    st.title("✈️ Make Predictions")
    st.subheader("Adjust the values below to make predictions on the Thyroid dataset:")

    # User inputs
    selected_age = st.number_input("Select Age", min_value=15, max_value=89, value=72)
    selected_gender = st.selectbox("Select Gender", gender)
    selected_country = st.selectbox("Select Country", countries)
    selected_ethnicity = st.selectbox("Select Ethnicity", ethnicities)
    selected_history = st.selectbox("Select Thyroid History", history)
    selected_exposure = st.selectbox("Select Radiation Exposure", exposure)
    selected_deficiency = st.selectbox("Select Iodine Deficiency", deficiency)
    selected_smoking = st.selectbox("Select Smoking", smoking)
    selected_obesity = st.selectbox("Select Obesity", obesity)
    selected_diabetes = st.selectbox("Select Diabetes", diabetes)
    selected_tsh = st.number_input("Select TSH levels", min_value=0.1, max_value=10.0, value=4.10)
    selected_t3 = st.number_input("Select T3 levels", min_value=0.5, max_value=3.5, value=3.3)
    selected_t4 = st.number_input("Select T4 levels", min_value=4.5, max_value=12.0, value=6.0)
    selected_nodule = st.number_input("Select Nodule size", min_value=0.0, max_value=5.0, value=1.25)
    selected_risk = st.selectbox("Select Risk level", risk)

    # Initialize user input dictionary with zeroes
    user_input = {col: 0 for col in X.columns}

    # Store numerical inputs
    user_input["age"] = selected_age
    user_input["tsh_level"] = selected_tsh
    user_input["t3_level"] = selected_t3
    user_input["t4_level"] = selected_t4
    user_input["nodule_size"] = selected_nodule

    # Encode binary and categorical variables
    user_input["gender"] = 1 if selected_gender == "Male" else 0
    user_input["family_history"] = 1 if selected_history == "Yes" else 0
    user_input["radiation_exposure"] = 1 if selected_exposure == "Yes" else 0
    user_input["iodine_deficiency"] = 1 if selected_deficiency == "Yes" else 0
    user_input["smoking"] = 1 if selected_smoking == "Yes" else 0
    user_input["obesity"] = 1 if selected_obesity == "Yes" else 0
    user_input["diabetes"] = 1 if selected_diabetes == "Yes" else 0

    # Encode risk level
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
    user_input["thyroid_cancer_risk"] = risk_mapping[selected_risk]

    # Set one-hot encoded country and ethnicity
    user_input[f"country_{selected_country}"] = 1
    user_input[f"ethnicity_{selected_ethnicity}"] = 1

    # Ensure user input follows the same column order as X
    user_df = pd.DataFrame([user_input], columns=X.columns)

    # Display the encoded user input
    st.write("### Your Input Values")
    st.dataframe(user_df)

    # Scale the training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input (ensure correct columns and order)
    user_input_scaled = scaler.transform(user_df)

    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_scaled, y)

    # Make a prediction
    prediction = model.predict(user_input_scaled)[0]

    # Map prediction to a readable label
    prediction_label = "Benign" if prediction == 0 else "Malignant"

    # Display the result
    st.subheader("Prediction Result:")
    st.write(f"🚀 The model predicts that the patient has **{prediction}** thyroid condition.")
