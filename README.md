# 🏡 California House Price Prediction
 
A machine learning project that predicts **median house values** across California districts using demographic and geographic features. Built with Python and
scikit-learn.


## 📌 Project Overview
 
This project trains a **Random Forest Regression** model on the California Housing dataset to predict house prices based on features like location, income, and 
population density. It uses a full scikit-learn pipeline for preprocessing and supports both **training** and **inference** modes automatically.


## 📂 Project Structure
 
california-house-price-prediction/
│
├── main.py               # Main script (train + inference)
├── main_old.py           # Experimental/comparison script
├── housing.csv           # Original dataset (20,641 rows)
├── input_data.csv        # Test data – 20% split (4,129 rows)
├── output_data.csv       # Predictions output
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation


## 📊 Dataset
 
**Source:** California Housing Dataset  
**Total Records:** 20,641  
**Test Split:** 4,129 rows (20%)
 
| Column | Description |
|--------|-------------|
| `longitude` | Geographic longitude of the district |
| `latitude` | Geographic latitude of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms in the district |
| `total_bedrooms` | Total number of bedrooms in the district |
| `population` | Total population of the district |
| `households` | Total number of households |
| `median_income` | Median income of households (in tens of thousands) |
| `median_house_value` | **Target variable** – Median house value (USD) |
| `ocean_proximity` | Proximity to the ocean (categorical) |


## 📊 Data Visualizations

### 1. Housing Prices by Location
<img width="610" height="438" alt="image" src="https://github.com/user-attachments/assets/14d8480f-229e-4e5e-8ee8-63839299b976" />

> The map plots California districts by latitude and longitude.
> Colors represent median house value — **blue = low prices**, 
> **red/yellow = high prices**.
> High-value homes cluster around **latitude 34-36 (Los Angeles area)**
> and **longitude -118 to -122 (coastal regions)**.
> Interior and northern regions show predominantly lower house values.


### 2. Scatter Matrix — Key Features
<img width="1022" height="785" alt="image" src="https://github.com/user-attachments/assets/0fcdf26f-0ac7-4f5d-90d5-10517920bd77" />

> **median_income vs median_house_value** → Strong positive 
> correlation — higher income areas have significantly higher 
> house prices.
> 
> **housing_median_age vs median_house_value** → Very weak 
> correlation — age of housing has little impact on price.
> 
> **housing_median_age** → Most houses are between 
> 15-35 years old (histogram peak).
> 
> **median_income** → Right-skewed distribution, most households 
> earn between $2,000-$6,000 (tens of thousands).
> 
> **median_house_value** → Capped at $500,000 — visible 
> as a flat line at the top of scatter plots.
 


## ⚙️ How It Works
 
The `main.py` script runs in two modes automatically:
 
### 🔧 Mode 1 — Training (First Run)
If `model.pkl` does not exist:
- Loads `housing.csv`
- Splits data using **Stratified Shuffle Split** (80% train / 20% test)
- Saves test data as `input_data.csv`
- Preprocesses features using a **scikit-learn Pipeline**
- Trains a **Random Forest Regressor**
- Saves model and pipeline as `.pkl` files
  
### 🔍 Mode 2 — Inference (Subsequent Runs)
If `model.pkl` exists:
- Loads saved model and pipeline
- Reads `input_data.csv`
- Makes predictions
- Saves results to `output_data.csv`

 
## 🛠️ Tech Stack
 
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Core programming language |
| pandas | Latest | Data loading, manipulation & exploration |
| NumPy | Latest | Numerical computations |
| scikit-learn | Latest | ML pipeline, preprocessing & modeling |
| joblib | Latest | Model serialization (save/load .pkl) |
| matplotlib | Latest | Data visualization & plotting |
| RandomForestRegressor | sklearn | Primary prediction model |
| StratifiedShuffleSplit | sklearn | Balanced train/test splitting |
| ColumnTransformer | sklearn | Parallel feature preprocessing |
| SimpleImputer | sklearn | Handling missing values |
| StandardScaler | sklearn | Numerical feature normalization |
| OneHotEncoder | sklearn | Categorical feature encoding |


 
## 🚀 Getting Started
 
### 1. Clone the Repository
```bash
git clone https://github.com/sitasaini001/California-House-Price-Prediction.git
cd California-House-Price-Prediction
```
 
### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib
```
 
### 3. Run the Script
```bash
python main.py
```
 
- **First run** → Trains the model and prints: `congrats! your model is trained.`
- **Subsequent runs** → Runs inference and prints: `Inference is completed`
- Output predictions saved to `output_data.csv`

 
## 📈 Model Details
 
| Parameter | Value |
|-----------|-------|
| Model | Random Forest Regressor |
| Train/Test Split | 80% / 20% |
| Split Strategy | Stratified Shuffle Split (by income category) |
| Numerical Preprocessing | Median Imputation + Standard Scaling |
| Categorical Preprocessing | One-Hot Encoding (`ocean_proximity`) |
| Evaluation Metric | RMSE (Root Mean Squared Error) |
 

 
## 📁 Output
 
After inference, `output_data.csv` contains all original test features plus a new column:
 
| Column | Description |
|--------|-------------|
| `median_house_value` | Predicted house price (USD) |
 

 
## 👩‍💻 Author
 
**Sita Saini**  
GitHub: [@sitasaini001](https://github.com/sitasaini001)
 
 
