# 🚬 Tobacco Use and Mortality (2004–2015)

## 📖 Overview
The **Tobacco Use and Mortality (2004–2015)** project analyzes health data to explore the relationship between tobacco consumption patterns and mortality rates across different regions and demographics.  
Using **Python**, **Pandas**, and **Machine Learning models**, this project investigates how smoking and tobacco-related habits contribute to overall mortality and public health outcomes.

The analysis is supported by a trained predictive model (`.pkl` file), which can be downloaded from Google Drive for further testing or reuse.

---

## 🎯 Objectives
- Perform **Exploratory Data Analysis (EDA)** on tobacco use and mortality data.  
- Identify **patterns and correlations** between tobacco consumption and death rates.  
- Visualize **age-wise, gender-wise, and region-wise** trends in tobacco impact.  
- Build a **regression model** to predict mortality risk based on tobacco use metrics.  
- Provide a downloadable `.pkl` model for researchers and analysts.

---

## 📊 Dataset
- **Source:** [CDC – Tobacco Use Data (2004–2015)](https://data.cdc.gov/) or [Kaggle Public Health Datasets](https://www.kaggle.com/)
- **File Format:** CSV (`tobacco_use_mortality.csv`)
- **Data Columns Include:**
  - `Year`
  - `Region`
  - `Gender`
  - `Age_Group`
  - `Tobacco_Use_Percentage`
  - `Mortality_Rate`
  - `Population`
  - `Education_Level`
  - `Income_Category`

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| **Programming Language** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Model Storage** | Pickle (`.pkl`) |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 🧾 Project Workflow

### 1️⃣ Data Preprocessing
- Load dataset and check for missing/null values.  
- Handle missing values and standardize column formats.  
- Convert categorical data (like Gender, Region) into numerical form.  
- Normalize or scale continuous variables for modeling.

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyze tobacco usage patterns by **year**, **gender**, and **region**.  
- Correlation analysis between **tobacco use %** and **mortality rate**.  
- Visualize key health indicators through bar plots, heatmaps, and trend lines.  
- Identify the **most affected age groups** and **high-risk regions**.

### 3️⃣ Predictive Modeling
- Train models such as:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
- Evaluate model performance using RMSE and R² metrics.  
- Save the final model as `tobacco_mortality_model.pkl`.

### 4️⃣ Visualization
- Yearly mortality trend vs tobacco usage.  
- Gender-based mortality rate visualization.  
- Regional comparison of smoking prevalence.  
- Predicted vs Actual mortality rate plot.

---

## 📦 Model File (PKL Download Section)

📁 **Download Pre-trained Model:**  
👉 [Click here to download `tobacco_mortality_model.pkl`](https://drive.google.com/uc?id=YOUR_FILE_ID&export=download)

> ⚠️ Replace `YOUR_FILE_ID` with your actual Google Drive file ID.  
> Example:
> ```
> https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQrStUvWxYz123456&export=download
> ```

