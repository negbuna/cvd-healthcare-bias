# Predicting Cardiovascular Risk with Fairness in Mind  
This notebook investigates how model performance and fairness vary across gender in predicting 10-year heart disease risk using the Framingham Heart Study dataset.

---

## Dataset  
I used the [Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset), which contains anonymized health data from 4,240 individuals, including:

- Demographics (e.g., age, sex, education)
- Health indicators (e.g., BMI, blood pressure, glucose)
- Outcome: `TenYearCHD` — whether the person developed coronary heart disease within 10 years.

---

## Model
I used scikit-learn to make a Logistic Regression model to predict 10-year CHD risk.

```python
lr_model = LogisticRegression(max_iter=2000)
```

Overall model accuracy: 86.2%

I then evaluated its fairness and error rates across gender groups.

### Fairness Metrics by Gender
I compared True Positive Rate (TPR) and Accuracy between males and females to identify disparities in the model’s ability to correctly identify at-risk patients.

![True Positive Rate and Accuracy](images/tpr_acc.png)

Equal Opportunity & Demographic Parity Differences
Equal Opportunity Difference (EOD): **0.078**
(Difference in TPR between male and female)

Demographic Parity Difference (DPD): **-0.09**
(Difference in positive prediction rates)

![Equal Opportunity & Demographic Parity Differences](images/eod_dpd.png)

### Confusion Matrices by Gender
These show how predictions broke down into true/false positives/negatives by gender.

Male

![Confusion Matrix - Male](images/conf_matrix_male.png)

Female

![Confusion Matrix - Female](images/conf_matrix_male.png)

### Key Findings
* Disparity in TPR: The model is better at detecting risk in men (17%) than in women (9.5%).
* The model is fairly accurate in classifying those with low risk for CHD, much more accurate than it is for assessing those at risk.
* Accuracy Gap: The model is more accurate for women (90%) than men (81%).
* This suggests possible bias due to representation or feature interaction, especially since education level and access to care (which correlate with SES and potentially race) may affect who gets diagnosed and included in the dataset.

### Future Work
* Apply rebalancing techniques like SMOTE or reIighting.
* Consider intersectional fairness (e.g., by age + gender or education + gender).
* Train alternative models and compare fairness trade-offs.
* Explore fairness-aware loss functions or post-processing methods.

---

## How to Run

1. Clone the repo and open the Colab notebook:
```git
git clone https://github.com/yourusername/your-repo-name.git
```
2. Open the notebook in Google Colab and install dependencies, download the dataset, then run the model.

## References

- Buolamwini, J., & Gebru, T. (2018). *Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification*. Proceedings of the 1st Conference on Fairness, Accountability and Transparency. [Link](https://proceedings.mlr.press/v81/buolamwini18a.html)

- Dunn, C., Correia, D., & Rathke, L. (2022). *Assessing Underrepresentation in Machine Learning Datasets for Cardiovascular Disease (CVD) Research*. Worcester Polytechnic Institute. [Link](https://digital.wpi.edu/concern/student_works/1v53k137g)
