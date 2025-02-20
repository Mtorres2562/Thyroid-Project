# Thyroid Cancer Predictions  

## Problem Statement  
This analysis simulates real-world thyroid cancer risk factors using a dataset from [Kaggle](https://www.kaggle.com/datasets/ankushpanday1/thyroid-cancer-risk-prediction-dataset/data), providing insights into key contributors to the disease and identifying regions where thyroid cancer is most prevalent. The goal is to determine what makes a patient high-risk and which personal health factors they should monitor to stay proactive about potential thyroid cancer concerns.  

## Data Dictionary  

| Column Name          | Description  |  
|----------------------|-------------|  
| Patient_ID          | Observation number |  
| Age                | Age of patient |  
| Gender            | Gender of patient (Female or Male) |  
| Country           | Identifies the country of origin of the patient |  
| Ethnicity         | Identifies the ethnic background |  
| Family History    | Identifies Yes or No for a family history of thyroid cancer |  
| Radiation Exposure | Identifies Yes or No if the patient has been exposed to radiation |  
| Iodine Deficiency  | Identifies Yes or No if the patient is deficient in iodine levels |  
| Smoking           | Identifies Yes or No if the patient smokes |  
| Obesity           | Identifies Yes or No if the patient is obese |  
| Diabetes         | Identifies Yes or No if the patient is diabetic |  
| TSH Levels       | Provides the value level of Thyroid-Stimulating Hormone (TSH) |  
| T3 Levels        | Provides the value level of Triiodothyronine (T3) |  
| T4 Levels        | Provides the value level of Thyroxine (T4) |  
| Nodule Size      | Provides the size of the nodule |  
| Thyroid Cancer Risk | Identifies the patient's risk of thyroid cancer as Low, Medium, or High |  
| Diagnosis        | Identifies whether the patient has thyroid cancer (Malignant or Benign) |  

## Executive Summary  

### Data Cleaning Steps  
This dataset required no significant cleaning aside from converting categorical variables to numerical ones to improve the final modeling process.  

### Key Visualizations  

#### Visualization 1: Country vs. Diagnosis Comparison  
This shows that India has a 12% higher rate of thyroid cancer compared to other countries. 

![Visualization 1]('data_images/visualization_1.PNG')

#### Visualization 2: Ethnicity vs. Diagnosis Comparison  
This shows that Asians have a higher percentage of thyroid cancer cases—about 8% more than the next highest ethnicity, Africans, who have a rate of 25%. 

![Visualization 2]('data_images/visualization_2.PNG')

#### Visualization 3: Family History vs. Diagnosis Comparison  
Having a family history of thyroid cancer increases the risk by 13%, compared to 19% for those with no family history. 

![Visualization 3]('data_images/visualization_3.PNG')

#### Visualization 4: Radiation Exposure vs. Diagnosis Comparison  
The data shows that exposure to radiation increases the risk of thyroid cancer by 11% compared to those with no exposure. 

![Visualization 4]('data_images/visualization_4.PNG')

#### Visualization 5: Iodine Deficiency vs. Diagnosis Comparison  
Having low iodine levels increases the risk of thyroid cancer by 10%, compared to an efficiency level of 21% in individuals with sufficient iodine.

![Visualization 5]('data_images/visualization_5.PNG')

#### Visualization 6: Correlations with Diagnosis  
We can now observe both positive and negative correlations. Thyroid cancer risk has the highest positive correlation at 0.37, followed by family history (0.14), Asian ethnicity (0.14), and Indian origin (0.11).

![Visualization 6]('data_images/visualization_6.PNG')

#### Visualization 7: K-Neighbors Classifier Confusion Matrix  
With `n_neighbors` set to 51, the model's performance improved, increasing True Negative predictions by 719 and True Positive predictions by 610.

![Visualization 7]('data_images/visualization_7.PNG')

## Model Performance  

### Model Selection  
The K-Nearest Neighbors (KNN) model was used in this project to predict a patient's risk of thyroid cancer. This analysis aims to raise awareness, helping patients take better care of themselves and proactively manage potential thyroid cancer concerns.  

| N_Neighbors | Train Score | Test Score |  
|-------------|------------|------------|  
| 51          | 0.8268     | 0.8280     |  

## Conclusions & Recommendations  
I have identified key features that can help patients determine their risk of thyroid cancer, including family history, country of origin, and ethnic background. These factors play a significant role in assessing risk levels. Further analysis reveals that individuals of Asian ethnicity, particularly those of Indian origin with a family history of thyroid cancer, have the highest risk factors.  

To gain deeper insights, I would like to gather more data on patients from India to better understand whether thyroid cancer in this group is primarily hereditary or influenced by other factors.
