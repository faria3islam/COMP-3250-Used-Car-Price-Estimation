# COMP-3250 Project - Data Driven Used Car Price Estimation 

| Name          |         Contribution          |
| ------------- | ----------------------------- |
| Peter Ciastek | TBD                           |
| Faria Islam   | TBD                           |
| Ion Kerchani  | TBD                           |
| Cody Taylor   | TBD                           |

# Abstract 
The objective of our project is to build a data driven used car price estimation model. This 
would be based on factors including mileage, age, condition, and more. We chose this 
problem domain as we believe there is a need to add more transparency in the used car 
market to benefit both buyers and sellers. We would achieve our goal through using the data 
analytics lifecycle and machine learning to clean, discover, visualize, and model a dataset of 
used car listings. One platform to help in achieving our aim is Power BI which we can use 
visualize patterns and build forecasting insights. Our end results would include a dashboard 
for inputting vehicle information and a model that outputs estimated car prices. 

# Problem/Domain 
In the used car market, prices are highly variable because of factors such as mileage, year, 
make, age, conditions, etc. This makes it difficult for both buyers and sellers to guage a fair 
market price. There is the issue of both over/under pricing and over/under paying. Our project 
is important because a data driven approach allows helps buyers make smarter purchasing 
decisions and sellers set more accurate and competitive prices. Our project would be an 
example of how data analytics could be applied to solve real world economic challenges 
through turning data into actionable information. 

# Motivation 
To summarize, our main motivations for taking on this problem, is that the used car market is 
large, competitive, and yet confusing. The biggest pain point for both buyers and sellers is 
price transparency. This problem in particular is interesting as it involves messy data and 
non-linear relationships. It’s challenging due to high variance in conditions, multiple factors, 
and sometimes missing car information. We still have motivation for approaching this 
problem as there are multiple groups that can benefit, mainly individual buyers and sellers, as 
well as dealerships, online marketplaces, and insurance companies. 

# Tech Stack 
Our tools include Power BI for dashboarding, Python libraries (Pandas, NumPy, Scikit-learn, Matplotlib) for data cleaning, feature engineering, modeling, and evaluation, Jupyter Notebooks for exploratory analysis, and a used-car dataset sourced from Kaggle.

# Project Scripts

The project now includes modular scripts for cleaner workflow and easier reproducibility:

1. `src/evaluation.py`
- Purpose: evaluate model predictions in a reusable and consistent way.
- Calculates core regression metrics such as MAE and RMSE (and R2).
- Supports comparison-ready output through a structured metrics table.

2. `src/main.py`
- Purpose: provide a one-command pipeline that runs cleaning, training, and prediction.
- Integrates all components from raw data processing to final price estimate.

3. `requirements.txt`
- Purpose: list required Python libraries for setup and reproducibility.

# Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run complete pipeline:

```bash
python src/main.py --skip-predict
```

Run complete pipeline with final prediction step:

```bash
python src/main.py
```
