# Credit Score with Scorecards Method - Credit Scoring and Analytics
---
### -  Introduction/Background
PinjamPintar is a financial institution with a deep commitment to financial services. PinjamPintar faces challenges in the lending decision-making process. This process involves manual assessments that are time-consuming and at risk of subjective bias. Not only that, the risk associated with a Non-Performing Loan (NPL) or unpaid debt is always a major concern for PinjamPintar.
One approach that can be taken is the use of credit scores or credit score prediction models. Credit scoring is a method of evaluating the credit risk of loan applications. Using historical data and statistical techniques, credit scoring tries to isolate the effects of various applicant characteristics on delinquencies and defaults.

### - Dataset Description
The dataset used 255,347 rows from 18 variables, but only uses a few variables that are relevant in influencing/evaluating loan default. The data is a collection of information that includes demographic, financial, and loan information. The data consists of :
- 15 predictors/potential characteristics		- 1 response/target variable

### - Scorecard Development
1. Explore Data
2. Handle Missing Value and Outliers
3. Check Correlation
4. Initial Characteristic Analysis
5. Statistical Measures
6. Check Logical Trend and Business/Operational Considerations
7. Design Scorecard
8. Choose Scorecard

** - Implementation Plan**
1. Setting Cutoff Score
2. Workflow Credit Process
<img src="img/workflow.png" width="1000"/>
3. Prediction Credit Score User Interface
<img src="img/1.png" width="1000"/>
<img src="img/1.png" width="1000"/>

### - Conclusion
The logistic regression model that has been adjusted by selecting the right cut-off score is able to predict the credit score well. The model was able to distinguish between good and bad applicants with a recall rate of 68%. The overall quality of the model was rated as good with an AUC of 0.74. This indicates that the model can be used to support credit decisions well.

