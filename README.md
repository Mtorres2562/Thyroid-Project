## Thyroid Cancer Predictions

## Problem Statement
This analysis simulates real-world thyroid cancer risk factors using a dataset from [Kaggle](https://www.kaggle.com/datasets/ankushpanday1/thyroid-cancer-risk-prediction-dataset/data), providing insights into key contributors to the disease and identifying regions where thyroid cancer is most prevalent.
The goal is to determine what makes a patient high-risk and which personal health factors they should monitor to stay proactive about potential thyroid cancer concerns.

## Data Dictionary

| Column Name | Description |
|-------------|-------------|
| Patient_ID     | Observation number |
| Age     | Age of patient |
| Gender     | Gender of patient from Female and Male |
| Country     |  Identifies the country of origin of patient |
| Ethnicity     | Identifies ethnic background |
| Family History    | Identifies Yes or No of Families history of thyroid cancer |
| Radiation Exposure     | Identifies Yes or No if have been exposed to Radiation |
| Iodine Deficiency    | Identifies Yes or No if Deficient in Iodine levels |
| Smoking    | Identifies Yes or No if patient smokes |
| Obesity    | Identifies Yes or No if patient is obese|
| Diabetes     | Identifies Yes or No if patient is diabetic |
| TSH levels    | Gives value level of Thyroid-Stimulating Hormone |
| T3 Levels    | Gives value level of Triiodothyronine (T3) |
| T4 Levels     |  Gives value level of Thyroxine (T4) |
| Nodule Size     | Gives value of size of Nodule |
| Thyroid Cancer Risk     | Identifies patients risk of throid cancer from Low, Medium or High |
| Diagnosis     | Identifies patients of thyroid cancer or not as Malignant or Benign |





## Executive Summary

### Data Cleaning Steps
This dataset required no cleaning aside from converting categorical variables to numerical ones to improve the final modeling process.  

### Key Visualizations


#### Visualization 1: [Country Diagnosis comparison]
[This shows that India has a 12% higher rate of thyroid cancer compared to other countries.]

![Visualization 1]('data_images/visualization_1.PNG')

#### Visualization 2: [Ethnicity Diagnosis Conparisons]
[This shows that Asians have a higher percentage of thyroid cancer cases, about 8% more than the next highest ethnicity, Africans, at 25%]

![Visualization 2]('data_images/visualization_2.PNG')

#### Visualization 3: [Family History Diagnosis Comparison]
[This shows that having a family history increases the risk of thyroid cancer by 13%, compared to 19% for those with no family history.]

![Visualization 3]('data_images/visualization_3.PNG')

#### Visualization 4: [Radiation Exposure diagnosis comparison]
[The data shows that exposure increases the risk of thyroid cancer by 11% compared to those with no exposure.]

![Visualization 4]('data_images/visualization_4.PNG')

#### Visualization 5: [Iodine Deficiency Diagnosis Comparison]
[Having low iodine levels increases the risk of thyroid cancer by 10%, compared to an efficiency level of 21% in individuals with sufficient iodine.]

![Visualization 5]('data_images/visualization_5.PNG')

#### Visualization 6: [Corrilations with Diagnosis]
[We can now observe both positive and negative correlations. Thyroid cancer risk has the highest positive correlation at 0.37, followed by family history (0.14), Asian ethnicity (0.14), and Indian origin (0.11).]

![Visualization 6]('data_images/visualization_6.PNG')

#### Visualization 7: [KNeighbors Classifier Confusion Matrix]
[With n_neighbors set to 51, the model's performance improved, increasing True Negative predictions by 719 and True Positive predictions by 610.]

![Visualization 7]('data_images/visualization_7.PNG')

## Model Performance

### Model Selection
The K-Nearest Neighbors (KNN) model was used in this project to predict a patient's risk of thyroid cancer. This analysis aims to raise awareness, helping patients take better care of themselves and proactively manage potential thyroid cancer concerns.

| N_Neighbors       | Train score     | Test Score       |
|-------------------|----------|----------|
| 51 N_Neighbors    | [0.8267719003498037]  | [0.8279954112049348]  |

## Conclusions/Recommendations
I have identified key features that can help patients determine their risk of thyroid cancer, including family history, country of origin, and ethnic background. These factors play a significant role in assessing risk levels. Further analysis reveals that individuals of Asian ethnicity, particularly those of Indian origin with a family history of thyroid cancer, have the highest risk factors. To gain deeper insights, I would like to gather more data on patients from India to better understand whether thyroid cancer in this group is primarily hereditary or influenced by other factors. 
