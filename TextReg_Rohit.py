import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import modlee
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

os.environ['MODLEE_API_KEY'] = "E1S58A6F4dUUBJEG02E1R1TG631i8b8E"

import modlee
if os.environ.get('MODLEE_API_KEY') is None:
    print("Module key not set")
else:
    modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))
    print("Module initialized")

print('Stage 1 Successful')

### Importing Dataset and creating a dataframe
file_path = './data/train_2023.csv'
df = pd.read_csv(file_path)

X = df.drop(['claim_number', 'fraud'],axis=1)
y = df['fraud']

### Splitting dataframe in Training and Testing Datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


### Imputing missing values

# Fill missing values for witness_present_ind with the most frequent value
most_frequent_value = df['witness_present_ind'].mode()[0]
X_train['witness_present_ind'].fillna(most_frequent_value, inplace=True)

# Fill missing values for age_of_vehicle with the median
median_age_of_vehicle = df['age_of_vehicle'].median()
X_train['age_of_vehicle'].fillna(median_age_of_vehicle, inplace=True)

# Fill missing values for claim_est_payout with the median
median_claim_est_payout = df['claim_est_payout'].median()
X_train['claim_est_payout'].fillna(median_claim_est_payout, inplace=True)

# Ensure no missing values for marital_status for robustness
most_frequent_marital_status = df['marital_status'].mode()[0]
X_train['marital_status'].fillna(most_frequent_marital_status, inplace=True)

# Verify that there are no missing values left
print(X_train.isnull().sum())

### Handling Outliers
# Setting up the upper and lower bound for the outliers in given columns
possible_outlier_features = ['age_of_driver', 'annual_income']

def impute_outliers_with_mean(df, possible_outlier_features):
  for feature in possible_outlier_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    # Finding the mean of the feature
    mean_value = df[feature].mean()

    # Replacing the values less than lower bound and greater than upper bound with the mean value
    df[feature] = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound), mean_value, df[feature])

impute_outliers_with_mean(X_train, possible_outlier_features)


### OneHotEncoding and Normalizing the Dataset with Standard Scaler
# Transforming claim_date to numerical feature
X_train['claim_day_of_year'] = pd.to_datetime(X_train['claim_date']).dt.dayofyear
X_val['claim_day_of_year'] = pd.to_datetime(X_val['claim_date']).dt.dayofyear

# Adding claim_day_of_week as a categorical feature
X_train['claim_day_of_week'] = pd.to_datetime(X_train['claim_date']).dt.day_name()
X_val['claim_day_of_week'] = pd.to_datetime(X_val['claim_date']).dt.day_name()

numerical_cols = ['age_of_driver', 'age_of_vehicle', 'marital_status', 'zip_code', 'safty_rating',
                      'witness_present_ind', 'annual_income', 'high_education_ind',
                      'policy_report_filed_ind', 'address_change_ind', 'past_num_of_claims', 'liab_prct',
                      'claim_est_payout', 'vehicle_price', 'vehicle_weight', 'claim_day_of_year']

categorical_cols = ['gender', 'living_status', 'accident_site',  'channel',
                        'vehicle_category', 'vehicle_color', 'claim_day_of_week']



# X = df.drop(['median_house_value'],axis=1)
# y = df['median_house_value']

# categorical_cols = ['ocean_proximity']  # Replace with your actual categorical columns
# numerical_cols = [col for col in X if col not in categorical_cols]

print('Stage 2 Successful')

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

### Preprocess features
X = preprocessor.fit_transform(X_train)
### Transform the validation data
X_val = preprocessor.transform(X_val)

print('Stage 3 Successful')

### Define custom dataset
class TextRegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

### Create datasets
train_dataset = TextRegressionDataset(X_train, y_train)
val_dataset = TextRegressionDataset(X_val, y_val)

### Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print('Stage 4 Successful')

recommender = modlee.recommender.from_modality_task(
    modality='text',
    task='classification',
)
recommender.fit(train_dataloader)
modlee_model = recommender.model
print(f"\nRecommended model: \n{modlee_model}")

print('Stage 5 Successful')

with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader
    )

print('Stage 6 Successful')

### Validation
trainer.validate(
    model=modlee_model,
    val_dataloaders=val_dataloader
)

print('Stage 7 Successful')

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")
artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = sorted(os.listdir(artifacts_path))
print(f"Saved artifacts: {artifacts}")

print('Code Successful!!')