<H3>ENTER YOUR NAME: Pavithra D</H3>
<H3>ENTER YOUR REGISTER NO. 212223230146 </H3>
<H3>EX. NO.1</H3>
<H3>DATE 18.11.2026 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/creditcard.csv")

# Display the first five rows
print("--- INITIAL DATAFRAME HEAD ---")
print(df.head())

# Check for missing values
print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())

# Check for duplicate rows
print("\n--- TOTAL DUPLICATED ROWS ---")
print(df.duplicated().sum())


```

```

# Separate features and target variable
X = df.drop(columns=["Class"])
Y = df["Class"]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create the scaled DataFrame
df_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=df.index
)

df_scaled["Class"] = Y

# Display the scaled data
print("--- SCALED DATAFRAME HEAD ---")
print(df_scaled.head())

# Convert features and labels into arrays
X = df_scaled.drop(columns=["Class"]).values
Y = df_scaled["Class"].values

# Display matrix dimensions
print("\n--- SPLIT MATRICES OVERVIEW ---")
print("Input Features (X):", X.shape)
print("Output Labels (Y):", Y.shape)

```

```
# Split the dataset into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Display training and testing features
print("--- TRAINING FEATURES (Xtrain) ---")
print(Xtrain)

print("\n--- TESTING FEATURES (Xtest) ---")
print(Xtest)

# Display target values
print("\n--- TRAINING TARGET VALUES (Ytrain) ---")
print(Ytrain)

print("\n--- TESTING TARGET VALUES (Ytest) ---")
print(Ytest)

# Display dataset dimensions
print("\n--- DATASET SPLIT SUMMARY ---")
print("Training features:", Xtrain.shape)
print("Testing features :", Xtest.shape)
print("Training labels  :", Ytrain.shape)
print("Testing labels   :", Ytest.shape)

```
## OUTPUT:
<img width="1500" height="532" alt="Screenshot 2026-11-13 133639" src="https://github.com/user-attachments/assets/ad2fb38e-ab25-4a36-a3b8-121fbb8677d4" />
<img width="782" height="730" alt="Screenshot 2026-11-13 133703" src="https://github.com/user-attachments/assets/7470cb5a-340b-44c0-9d85-b95c6080337b" />
<img width="1046" height="590" alt="Screenshot 2026-11-13 133728" src="https://github.com/user-attachments/assets/8aa4bb4d-98ad-4a95-a124-bf0d1fa29261" />
<img width="952" height="637" alt="Screenshot 2026-11-13 133746" src="https://github.com/user-attachments/assets/86080236-cb14-4856-9391-8cb37958506a" />
<img width="690" height="247" alt="Screenshot 2026-11-13 133801" src="https://github.com/user-attachments/assets/e734f84c-0b5a-4156-b4e0-a14bacfe833e" />





## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


