# Bayesian-ML
https://github.com/LunaLiu-world/Bayesian-ML/issues/2#issue-2908799525

## Business statement
Lending platforms connect borrowers seeking loans with investors willing to fund these loans, making the accurate prediction of interest rates crucial for both parties. Borrowers need fair rates that reflect their risk profile, while investors require appropriate returns to compensate for the risk they assume. The platform itself needs accurate interest rate models to maintain profitability, competitiveness, and regulatory compliance. In this project, we analyze lending data containing various borrower characteristics, loan attributes, and credit history metrics to build models that can accurately predict interest rates. Such predictions help lending platforms maintain a balanced marketplace where risk is appropriately priced, leading to sustainable business operations and customer satisfaction.

## Methodology Introduction
Our analytical approach employs a comprehensive Bayesian framework to model the dependencies among borrower variables and generate personalized risk classifications.  The methodology begins with exploratory data analysis to understand variable distributions and basic correlations between different loan and borrower attributes.  We then develop a Bayesian Network to model the complex interdependencies between borrower characteristics, followed by Maximum A Posteriori (MAP) estimation for risk classification with carefully considered prior assumptions.  To address borrower heterogeneity, we implement Hierarchical Models for personalized risk estimation based on borrower segments.
The framework is supplemented with several comparative modeling approaches, including Bayesian linear regression, Dirichlet multinomial regression, and multinomial logistic regression.  For parameter estimation in complex scenarios, we employ Markov Chain Monte Carlo (MCMC) methods.  Model performance will be rigorously evaluated using multiple metrics including RMSE, MAE, MSE, and R-squared to ensure reliable default risk predictions that can effectively guide lending decisions and optimize the risk-return profile of the loan portfolio. 


## Exploratory Data Analysis 
To ensure the reliability and accuracy of our predictive model for loan interest rates, we conducted an extensive exploratory data analysis and data cleaning process. The dataset consists of various borrower attributes, loan details, and credit history indicators, some of which required transformation, imputation, or removal due to inconsistencies, excessive missing values, or irrelevance to the prediction task.
1) Data Structure
The dataset contains multiple categorical and numerical features, each describing different aspects of a loan applicant’s profile. We started by summarizing the dataset, identifying missing values, unique value counts, and data types. Using metadata, we mapped each feature to its definition, helping us make informed decisions during cleaning and preprocessing.
2) Feature Removal
Certain features were removed due to redundancy, excessive missing values, or lack of predictive utility. Unique identifiers such as X2 (Loan ID) and X3 (Borrower ID) were removed as they serve only as unique identifiers and provide no meaningful information for predicting interest rates. Features with a high proportion of missing values, such as X8 (Loan Grade), X9 (Loan Subgrade), X12 (Home Ownership), X16 (Loan Purpose), X25 (Months Since Last Delinquency), and X26 (Months Since Last Public Record), were dropped because a large portion of their values were missing, making imputation unreliable. Some categorical features, such as X10 (Employer Name), X18 (Loan Title), and X19 (First Three Digits of Zip Code), had too many unique values with little standardization, making them unsuitable for predictive modeling.
3) Handling Missing Values
We applied different imputation techniques to maintain data consistency and avoid introducing biases. Missing values in categorical features such as X7 (Number of Payments), X11 (Years Employed), X13 (Annual Income), X14 (Income Verification), X17 (Loan Category), X20 (State), and X32 (Initial Listing Status) were filled with the most frequent value (mode). Missing values in numerical features such as X21 (Debt-to-Income Ratio), X22 (Delinquencies in Last Two Years), X24 (Credit Inquiries in Last Six Months), X27 (Number of Open Credit Lines), X28 (Number of Derogatory Public Records), X29 (Total Revolving Balance), X30 (Revolving Line Utilization Rate), and X31 (Total Credit Lines) were imputed using the median to minimize the influence of outliers. Essential feature completeness was ensured by removing rows missing values in crucial categorical features such as X7 (Number of Payments) and X14 (Income Verification), since these variables are essential in determining loan terms and applicant credibility. Rows where the target variable X1 (Interest Rate) was missing were removed from the training dataset to ensure the model only learns from complete records.
4) Data Type Conversions
Percentage and currency values such as X1 (Interest Rate), X4 (Loan Amount Requested), X5 (Loan Amount Funded), X6 (Investor-Funded Portion), and X30 (Revolving Utilization Rate) were converted into numerical format after removing non-numeric characters such as percent, dollar signs, and commas. Date variables such as X15 (Loan Issued Date) and X23 (Earliest Credit Line Opened) were transformed into datetime format for potential feature engineering related to time-based trends.

## Feature Engineering
After cleaning the dataset, we performed feature engineering to enhance the predictive power of our model. This process involved encoding categorical variables, creating new interaction features, and standardizing numerical variables to improve model stability and performance.
1) Transforming Categorical Features
Categorical variables were transformed using label encoding. For binary categorical features, we applied LabelEncoder, while multi-category variables were converted into numerical codes using the .astype('category').cat.codes method. This approach ensured compatibility with machine learning models that require numerical input.
2) Create New Features
Days Between X15 and X23: This feature captures the time gap between two key loan-related dates, providing insights into borrower behavior over time.
Funded to Requested Ratio: Calculated as the proportion of the loan amount funded relative to the requested amount, this ratio can indicate the lender’s confidence in approving loans.
Loan to Income Ratio: By dividing the loan amount by the borrower's annual income, this feature reflects the financial burden of the loan relative to the borrower's earnings.
Active Credit Line Ratio: This ratio measures the number of active credit lines relative to total credit lines, providing insight into credit utilization and financial responsibility.
3) Standardization
To ensure that numerical features contributed equally to the model, we applied standardization using StandardScaler. This transformation helps prevent features with larger magnitudes from dominating the learning process. We standardized all continuous numerical features while excluding categorical variables and the target variable (X1).
<img width="610" alt="Image" src="https://github.com/user-attachments/assets/33c6b615-89d9-4524-b95f-d3334e31cd58" />


## Model
#### Linear regression model 
##### Description: 
Parameters: Uses standard OLS (Ordinary Least Squares) estimation
Common Use Cases: Baseline predictive model for interest rate prediction in loan analysis, widely used for its simplicity and interpretability
##### Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
##### Results
* Test MSE: 10.4491
* Test R²: 0.4609
* Test MAE: 2.5867
* Test RMSE: 3.2325

<img width="563" alt="Image" src="https://github.com/user-attachments/assets/a74fff5d-a3c1-458f-82c7-93787b0b70c4" />

<img width="663" alt="Image" src="https://github.com/user-attachments/assets/78b69db4-ce8f-41c1-b741-03dbcf8db309" />

#### Bayesian linear regression model  
##### Description:
Parameters: Incorporates prior distributions for model parameters, combines prior knowledge with observed data
Common Use Cases: Credit risk modeling where uncertainty quantification is important
##### Formula: P(β|X,y) ∝ P(y|X,β) × P(β)
##### Results
- Test MSE: 11.4481
- Test R²: 0.4093
- Test MAE: 2.7086
- Test RMSE: 3.3836
- Feature Importance: Number_of_Payments (0.42), Revolving_Utilization_Rate (0.41), Credit_Inquiries_6M (0.23)







