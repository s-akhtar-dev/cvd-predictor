# AI-Driven Prediction of Cardiovascular Mortality and Risk Using Multi-Source U.S. Healthcare Data

**Author:** Sarah Akhtar  
**Department:** Computer Science, University of the Pacific  
**Location:** Stockton, CA, USA  
**Email:** s.akhtar@pacific.edu  
**Project Type:** Final Project Report  

---

## Abstract

Cardiovascular disease (CVD) remains the leading cause of mortality worldwide, responsible for approximately 17.9 million deaths annually. Traditional risk assessment methods often require extensive laboratory testing, imaging, and specialist consultation, limiting scalability and accessibility. This project, CVD Predictor, presents an AI-driven system that leverages multi-source U.S. healthcare datasets to predict cardiovascular mortality and assess patient risk. By integrating structured epidemiological datasets, synthetic risk factors, and enhanced data augmentation, we develop an ensemble machine learning model achieving 90.2% accuracy. A real-time dashboard interface supports visualization of risk predictions, patient cohorts, and population-level insights, demonstrating the feasibility of scalable, data-driven cardiovascular risk prediction.

**Index Terms:** Cardiovascular Disease, Machine Learning, Mortality Prediction, Risk Assessment, Data Integration, Ensemble Models

---

## 1. Introduction

### Background and Motivation
Cardiovascular disease (CVD) represents one of the leading causes of morbidity and mortality worldwide, encompassing a wide spectrum of conditions such as coronary artery disease, hypertension, myocardial infarction, and stroke. According to the World Health Organization, an estimated 17.9 million people die each year from CVD, accounting for nearly one-third of all global deaths. Despite ongoing advancements in healthcare delivery, early detection, and treatment, CVD continues to impose profound public health, social, and economic burdens on both developed and developing nations.

Risk prediction and stratification are essential for timely interventions, as they allow clinicians and policymakers to identify high-risk populations and tailor preventive strategies. However, existing risk assessment frameworks—such as the Framingham Risk Score and pooled cohort equations—often rely on invasive laboratory testing, imaging, or specialist consultations, making them resource-intensive and less accessible in low-resource healthcare settings. Furthermore, these models are typically constrained by demographic biases, limited generalizability across diverse populations, and challenges in incorporating heterogeneous data sources.

Recent advances in artificial intelligence (AI) and machine learning (ML) present new opportunities to overcome these limitations. ML-based models can handle high-dimensional, nonlinear, and heterogeneous datasets, thereby enabling more robust and scalable predictive systems for CVD. Moreover, the use of augmented and synthetic datasets has shown potential in addressing class imbalance, enhancing model generalizability, and expanding training data without requiring additional costly data collection.

### Problem Statement
This project investigates whether mortality records, clinical risk prediction datasets, and augmented synthetic health data can be combined into a unified AI-driven system for predicting cardiovascular mortality risk. Our approach integrates multiple data sources of varying granularity: (1) national mortality and epidemiological statistics, (2) structured clinical datasets with demographic and health risk features, and (3) synthetic and augmented datasets designed to expand sample diversity.

### Approach Overview
We implement and compare multiple machine learning models—including logistic regression, random forests, and gradient boosting—while leveraging an ensemble approach to combine their strengths and maximize predictive performance. By designing a scalable, data-driven system, this project aims to demonstrate the feasibility of multi-source CVD risk prediction while addressing critical gaps in accessibility, robustness, and interpretability. Ultimately, the goal is to provide a framework that can be adapted for real-world applications in both clinical and public health contexts, with potential to inform early interventions and reduce global cardiovascular mortality.

---

## 2. Related Work

### Academic Foundation
Research on cardiovascular disease (CVD) risk prediction spans several decades, beginning with traditional statistical models and evolving toward machine learning and deep learning approaches. Early work laid the foundation for population risk assessment, while more recent studies have explored interpretability, causal inference, and synthetic data generation to enhance reliability and scalability.

**Framingham Heart Study: General Cardiovascular Risk Profile**
D'Agostino et al. introduced the General Cardiovascular Risk Profile within the Framingham Heart Study. As one of the most influential statistical models, it provided physicians with a standardized tool to estimate long-term cardiovascular risk based on clinical and demographic variables. Its limitations include reliance on linear assumptions, fixed covariates, and limited external generalizability.

**World Health Organization: Cardiovascular Diseases (CVDs)**
The World Health Organization (WHO) published global fact sheets estimating 17.9 million annual deaths due to CVD worldwide. These reports highlight the severity and global distribution of CVD, emphasizing the disproportionate burden in low- and middle-income countries and calling for scalable, preventive solutions.

**Centers for Disease Control and Prevention: Heart Disease Facts**
The Centers for Disease Control and Prevention (CDC) compile statistics on cardiovascular outcomes across the U.S. population. The CDC emphasizes disparities in prevalence, mortality, and modifiable risk factors across socioeconomic and demographic subgroups, providing critical context for risk modeling.

**American Heart Association: Heart Disease and Stroke Statistics — 2022 Update**
The American Heart Association publishes annual epidemiological updates on heart disease and stroke. The 2022 update underscores that CVD remains the leading cause of death globally, while also highlighting improvements in prevention and treatment. These reports inform ongoing priorities for predictive and preventive modeling.

**Deep Learning for Healthcare: Miotto et al.**
Miotto et al. authored Deep Learning for Healthcare: Review, Opportunities and Challenges, highlighting the promise of deep learning for uncovering complex, nonlinear disease patterns. They also identified challenges in data quality, interpretability, and deployment in sensitive healthcare settings.

**Deep EHR: Shickel et al.**
Shickel et al. conducted a survey, Deep EHR: A Survey of Recent Advances on Deep Learning Techniques for Electronic Health Record (EHR) Analysis. They classified architectures such as recurrent neural networks, CNNs, and autoencoders, demonstrating their applications in diagnosis, prognosis, and treatment support, while noting persistent issues of heterogeneity and explainability.

**Scalable Deep Learning with Electronic Health Records: Rajkomar et al.**
Rajkomar et al., in their work Scalable and Accurate Deep Learning with Electronic Health Records, showed that deep learning models trained on millions of EHR samples could achieve state-of-the-art accuracy. Their work exemplified the scalability of DL methods, though it also reinforced the need for clinically interpretable outputs.

**Improving Accuracy with Causal Machine Learning: Richens et al.**
Richens et al. introduced causal machine learning in medicine in their article, Improving the Accuracy of Medical Diagnosis with Causal Machine Learning. By focusing on cause–effect modeling rather than associative patterns, their framework mitigates confounding, thus improving robustness and interpretability—both critical for high-stakes clinical applications.

**Efficient Multi-Scale 3D CNN with Conditional Random Fields: Kamnitsas et al.**
Kamnitsas et al. presented Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation. While centered on neuroimaging, this study showcased techniques for addressing class imbalance, leveraging context-aware features, and integrating multiple scales—approaches applicable to broader healthcare AI challenges.

**Data Augmentation for Cardiovascular Risk Prediction: Ghodsi et al.**
Ghodsi et al. explored Data Augmentation Approaches for Improving Cardiovascular Risk Prediction Models. Their results showed that synthetic augmentation can mitigate small-sample and imbalance issues common in clinical datasets, ultimately improving generalization and fairness—key considerations for building scalable CVD predictive frameworks.

### How This Work Builds Upon and Differs from Existing Work
While prior research has laid substantial groundwork for cardiovascular disease risk prediction, our project advances the field in several important ways:

1. **Data Scale and Integration:** Unlike prior work that focused on single datasets or limited cohorts (e.g., 70,000 samples in the commonly used Kaggle cardiovascular dataset), our framework integrates over 800,000 samples across five curated datasets, including risk factor-based clinical records and mortality trends. This large-scale integration improves generalizability and reduces demographic biases common in smaller studies.

2. **Novel Normalization Technique:** While past models often output raw probabilities or uncalibrated scores that may not be clinically interpretable, our project introduces a normalization method that transforms risk outputs into user-friendly, clinically intuitive ranges (e.g., 15–25% low risk, 45–55% moderate risk, 70–80% high risk) while preserving ordering accuracy. This directly addresses the interpretability concerns noted by Miotto et al., Shickel et al., and Rajkomar et al.

3. **Real-Time Clinical Deployment:** Whereas many previous approaches remain in research or simulation settings, our project deploys the predictive system as a production-ready web application using Flask. The system provides instant predictions at the point of care alongside actionable recommendations, bridging the gap between model development and practical clinical decision support.

4. **Comprehensive Validation:** Building upon the performance reporting standards in studies like Weng et al. (2017) and traditional epidemiological validation, our project implements 10-fold cross-validation, multiple performance metrics (accuracy, precision, recall, F1, ROC-AUC), and error range estimation to ensure statistical robustness. Achieving 90.29% accuracy with stable variance demonstrates reliability suitable for clinical environments.

---

## 3. Implementation Details

### Methodology
The methodology underpinning this cardiovascular disease risk prediction system integrates rigorous data processing, feature engineering, model development, validation, and deployment strategies. Its design focuses on accuracy, clinical interpretability, and scalability.

**a) Data Harmonization and Processing:** The five heterogeneous datasets were harmonized by standardizing variable names, scales, and units. Missing values were imputed, and outliers removed based on domain heuristics. Synthetic minority oversampling was applied to address class imbalance, ensuring adequate representation of less frequent cardiovascular mortality events without inflating noise.

**b) Feature Engineering Strategy:** Initially, eleven features were selected based on clinical relevance as reported in the literature: age, sex, BMI, smoking status, alcohol consumption, diabetes status, physical activity, cholesterol, systolic and diastolic blood pressure, and blood glucose. Correlation analyses and feature importance scores from preliminary Random Forest models guided refinement, confirming the exclusion of redundant or weakly predictive variables. This approach optimized predictive power and model transparency, supporting clinical usability.

**c) Modeling Framework:** Random Forest classification was selected for its demonstrated robustness and interpretability in medical prediction tasks. Hyperparameter tuning was conducted using grid search over parameters including number of estimators, maximum tree depth, and minimum samples per leaf. Cross-validation performance metrics optimized model selection, striking a balance between underfitting and overfitting.

To enhance the clinical interpretability of probabilistic outputs, a normalization function maps raw model predictions into risk categories—low, moderate, and high—to match familiar clinical risk stratification ranges, facilitating clearer communication with healthcare practitioners.

**d) Validation and Evaluation:** To guarantee generalizability, we employed stratified 10-fold cross-validation during model development. Evaluation metrics included accuracy, precision, recall, F1-score, and ROC-AUC, collected on training, validation, and test datasets. Confusion matrices and error analyses identified misclassification trends, guiding iterative model refinements.

**e) Deployment:** The final model is deployed via a Flask web application featuring real-time input validation, instant prediction generation, and interactive visualizations to assist practitioners in risk communication.

### Data Sources and Preprocessing
To build a robust and generalizable predictive model, we integrated five major cardiovascular datasets totaling over 800,000 individual patient samples:

1. **CardioTrain Dataset (70,000 samples):** The foundational clinical dataset containing 11 well-established cardiovascular risk factors, including demographic, clinical, and behavioral variables.

2. **Heart Disease Dataset (319,000 samples):** A large-scale collection capturing a broad spectrum of cardiovascular health indicators and outcomes, providing diversity in patient demographics and conditions.

3. **CVD Cleaned Dataset (308,000 samples):** A rigorously preprocessed dataset optimized for consistency and removal of noise and missing values, boosting model reliability.

4. **Risk Dataset (308,000 samples):** Population cardiovascular risk data used to enhance model stratification capabilities.

5. **Synthetic Data (50,000 samples):** Augmented data synthetically generated to address class imbalance and enrich minority classes.

6. **Combined Cardiovascular Data:** A unified dataset merging all above sources, characterized by over 855,000 total samples to maximize model generalizability and robustness.

All datasets underwent careful cleaning and standardization, including:
- **Missing Value Imputation:** Missing clinical readings were imputed using median or mode values as appropriate to preserve data integrity without distorting distribution.
- **Outlier Detection and Removal:** Extreme values inconsistent with physiological plausibility were identified using interquartile range thresholds and removed to prevent model bias.
- **Feature Engineering:** Eleven key cardiovascular risk factors were extracted or computed, including age, sex, body mass index (BMI), smoking status, alcohol consumption, diabetes status, physical activity, cholesterol levels, systolic and diastolic blood pressure, and fasting glucose.
- **Class Imbalance Handling:** Given the inherent imbalance of cardiovascular event outcomes, data augmentation techniques including synthetic generation of minority class examples were applied following best practices from prior literature. This improved model sensitivity without sacrificing specificity.

### Tools, Frameworks, and Programming Languages
The project leveraged a state-of-the-art technology stack optimized for data science and web deployment:

**Core Technologies:**
- **Python 3.8+:** The core programming language used for all project implementation. Python's rich ecosystem supports rapid development and deployment of machine learning solutions for healthcare, with extensive libraries for data manipulation, modeling, and visualization.
- **Scikit-learn 1.0+:** A mature and widely-used machine learning library that provided the implementation for Random Forest classifiers. It offers robust tools for model training, hyperparameter tuning, evaluation, and feature importance extraction—critical for building interpretable clinical models.
- **Flask 2.0+:** A lightweight web framework used to transform the trained model into a RESTful API, facilitating real-time predictions in clinical environments. Flask's simplicity and extensibility allow seamless integration of the prediction pipeline and interactive visualization frontends.
- **Pandas and NumPy:** Essential Python libraries enabling efficient handling, cleaning, and transformation of large cardiovascular datasets. Pandas offers versatile DataFrame structures for tabular data, while NumPy supports high-performance numerical operations crucial for preprocessing pipelines.
- **Joblib:** Utilized for fast and memory-efficient serialization of trained models, enabling swift loading and unloading during web application runtime. This accelerates prediction response times and reduces resource consumption.
- **Plotly:** Provides interactive, dynamic visualization components within the application interface, empowering clinicians to explore individualized risk predictions and feature impacts through user-friendly charts and graphs.
- **Google Colab:** Cloud-based Jupyter notebook environment used for model prototyping, training, and experimentation. Google Colab offers free access to GPU and TPU resources, facilitating efficient model training on large-scale datasets without local hardware constraints.

Development was conducted in a Unix shell environment on macOS, employing Git alongside GitHub for version control and collaborative code management. Python virtual environments ensured reproducible dependency handling across development and deployment phases.

### Testing Environment and Workflows
Rigorous validation and testing procedures were implemented to ensure the model's accuracy, stability, and clinical reliability:

**10-Fold Cross Validation:** The entire dataset was partitioned into ten stratified folds, ensuring each fold reflected the overall distribution of cardiovascular events and risk factors. Each fold was used once as a validation set while the other nine served as training data. This procedure was repeated for all folds, and performance metrics were averaged, minimizing bias from any specific data split and giving a robust estimate of model stability.

**Comprehensive Performance Metrics:** To evaluate predictive quality, these metrics were calculated:
- **Accuracy:** Proportion of correct predictions out of total cases.
- **Precision:** Ratio of true positive predictions to all positive predictions, reflecting false positive rate.
- **Recall (Sensitivity):** Ratio of true positive predictions to actual positives, indicating model's ability to detect high-risk cases.
- **F1-Score:** Harmonic mean of precision and recall, balancing false positives and false negatives.
- **ROC-AUC:** Area under the receiver operating characteristic curve, summarizing model's discrimination power at varying thresholds.

**Data Partitioning Strategies:** Strict segregation of datasets was maintained—no overlap occurred between training, validation, and testing stages. This prevented information leakages that artificially inflate performance and ensured the model's generalizability to unseen data.

**Error and Stability Analysis:** Variance and standard deviation of accuracy and AUC scores across cross-validation folds were measured. Minimal deviations (approximately ±0.5%) indicated strong model stability and reduced sensitivity to data sampling variability.

**Realistic Clinical Scenario Testing:** Beyond statistical evaluation, the deployed application was subjected to clinical scenario simulations including edge cases and typical patient profiles. This testing verified logical consistency and meaningful output, strengthening confidence in real-world use.

### Performance Results

**Summary of 10-Fold Cross-Validation Results and Key Performance Metrics:**

| Metric | Mean (%) | Standard Deviation (%) |
|--------|----------|------------------------|
| Accuracy | 90.29 | 0.52 |
| Precision | 91.2 | 0.47 |
| Recall (Sensitivity) | 89.8 | 0.55 |
| F1-Score | 90.5 | 0.49 |
| ROC-AUC | 94.3 | 0.38 |

These results confirm that the model consistently achieves high accuracy and balanced precision–recall tradeoffs, with excellent discrimination ability as evidenced by the ROC-AUC above 94%. Low variance across folds demonstrates reliable stability, making it suitable for clinical risk prediction tasks.

### Model Deployment and Architecture
The prediction system was structured as a modular Flask web application designed specifically for clinical settings. Key architectural features include:

- **Input Validation and Sanitization:** User inputs are rigorously checked to ensure valid ranges and data types, minimizing risks from erroneous data entry.
- **Prediction Pipeline:** Inputs pass through preprocessing, feature scaling, model inference using the Random Forest classifier, and output normalization for clinical interpretability.
- **Interactive Visualization:** Real-time drawing of patient-specific risk charts, comparing individual risk percentages against population averages.
- **Fast Response Times:** The model delivers predictions with minimal latency, supporting real-time clinical decision support requirements.

### Sample Inputs and Outputs

**Low-Risk Patient Input:** A 28-year-old non-smoking female (sex: 0), standing 165 cm tall and weighing 58 kg (BMI: 21.3), reports moderate alcohol consumption (3 units/week) and engages in regular physical activity. Her clinical readings include a cholesterol level of 170 mg/dL, systolic blood pressure of 120 mmHg, diastolic blood pressure of 80 mmHg, glucose level of 100 mg/dL, and no diabetes diagnosis.

**Low-Risk Patient Output:**
- Predicted risk score: 18% (Low Risk)

**High-Risk Patient Input:** A 68-year-old male (sex: 1), measuring 170 cm in height and 88 kg in weight (BMI: 30.4), is a smoker with high alcohol intake (18 units/week) and no regular physical activity. Clinical indicators show a cholesterol level of 280 mg/dL, systolic blood pressure of 160 mmHg, diastolic blood pressure of 95 mmHg, glucose level of 120 mg/dL, and a diagnosis of diabetes.

**High-Risk Patient Output:**
- Predicted risk score: 73% (High Risk)

**Key Notes from Actual Testing:**
- **Exact Input Field Names:** All input keys must be named precisely as follows: `sex`, `age_years`, `height_cm`, `weight_kg`, `bmi`, `smoking`, `alcohol_consumption`, `physical_activity`, `cholesterol`, `systolic_bp`, `diastolic_bp`, `glucose`, and `diabetes`. Incorrect field names will result in processing errors or failed predictions.
- **Output Format:** The system returns two fields: `risk_category` (classified as Low, Moderate, or High) and `risk_probability`, a decimal between 0 and 1 representing the predicted probability of cardiovascular risk.
- **Risk Category Thresholds:** Risk classification is determined based on the following probability thresholds:
  - Low: ≤30%
  - Moderate: 30%–65%
  - High: >65%
- **Visual Outputs:** The output includes comparative visualizations, enabling users to interpret patient results.

---

## 4. Conclusion and Summary

### Key Findings and Contributions
This work demonstrates the feasibility and efficacy of leveraging diverse multi-source healthcare data to develop a scalable, interpretable, and clinically relevant machine learning system for cardiovascular mortality prediction. By integrating over 855,000 patient records across multiple clinical and synthetic datasets, the developed Random Forest ensemble achieved high predictive accuracy, with cross-validated metrics exceeding 90%. The feature importance rankings align well with established cardiovascular risk factors, enhancing trust and interpretability—a critical aspect for clinical adoption.

A key contribution of this project lies in the successful fusion of heterogeneous datasets that span demographic, clinical, and epidemiological domains. This broad data foundation substantially improves model generalizability and minimizes biases inherent in single-source or narrow cohorts. The novel normalization approach facilitates translation of raw model outputs into clinically intuitive risk ranges, empowering healthcare providers to make timely, evidence-based decisions.

### Future Directions
Beyond traditional machine learning techniques, we recognize the emerging importance of causal machine learning methods in cardiovascular risk prediction. Unlike purely associative models, causal methods aim to illuminate the underlying cause-effect relationships, enabling more precise identification of modifiable risk factors and tailored intervention strategies at an individual level. Integrating causal inference frameworks can improve robustness against confounding and bias, ultimately enhancing the reliability of clinical recommendations. Future iterations of this system will explore incorporating causal forest algorithms and explainable AI frameworks, following recent advances demonstrating improved decision support in cardiovascular care.

Despite these strengths, limitations exist. The current model focuses on eleven key variables and utilizes retrospective data; integration of additional biomarkers, genetic information, and real-time patient monitoring data could further elevate predictive capabilities. Moreover, prospective validation studies are necessary to establish clinical impact and assess integration challenges in operational healthcare environments. Temporal modeling approaches—such as longitudinal deep learning or survival analysis—could complement existing static prediction frameworks by capturing disease progression dynamics.

Future work will also investigate expanding multi-modal data inputs, causal modeling integration, and deployment within electronic health record systems with continuous learning capabilities. Emphasizing transparency, fairness, and clinical usability, these efforts strive to translate predictive modeling advances into tangible health outcomes, ultimately aiding in reducing the burden of cardiovascular disease globally.

### Long-term Vision
In summary, this project both validates that leveraging large, diverse data and interpretable machine learning models can accurately predict cardiovascular mortality risk and charts a path for future hybrid, causal-informed, and scalable clinical decision support solutions. The demonstrated framework advances precision cardiology and sets the stage for innovative, data-driven personalized preventive care.

---

## 5. References

[1] World Health Organization, "Cardiovascular diseases (CVDs)," 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds).

[2] Centers for Disease Control and Prevention, "Heart Disease Facts," 2023. [Online]. Available: https://www.cdc.gov/heartdisease/facts.htm.

[3] R. B. D'Agostino, R. S. Vasan, M. J. Pencina, P. A. Wolf, M. Cobain, J. M. Massaro, and W. B. Kannel, "General cardiovascular risk profile for use in primary care: The Framingham Heart Study," Circulation, vol. 117, no. 6, pp. 743–753, 2008.

[4] American Heart Association, "Heart Disease and Stroke Statistics — 2022 Update," Circulation, vol. 145, no. 8, pp. e153–e639, 2022.

[5] J. G. Richens, C. M. Lee, and S. Johri, "Improving the accuracy of medical diagnosis with causal machine learning," Nature Communications, vol. 11, no. 1, p. 3923, 2020.

[6] B. Shickel, P. J. Tighe, A. Bihorac, and P. Rasheed, "Deep EHR: A survey of recent advances on deep learning techniques for electronic health record (EHR) analysis," IEEE Journal of Biomedical and Health Informatics, vol. 22, no. 5, pp. 1589–1604, 2018.

[7] R. Miotto, F. Wang, S. Wang, X. Jiang, and J. T. Dudley, "Deep learning for healthcare: review, opportunities and challenges," Briefings in Bioinformatics, vol. 19, no. 6, pp. 1236–1246, 2016.

[8] A. Rajkomar, E. Oren, K. Chen, A. M. Dai, N. Hajaj, M. Hardt, P. Liu, X. Liu, J. Marcus, M. Sun, P. Sundberg, H. Yee, K. Zhang, Y. Zhang, E. Shafran, P. J. Dean, J. F. Kelly, N. K. Maisog, J. Q. Chang, G. E. Dudley, and J. Dean, "Scalable and accurate deep learning with electronic health records," npj Digital Medicine, vol. 2, p. 18, 2019.

[9] K. Kamnitsas, C. Ledig, V. F. Newcombe, J. P. Simpson, A. D. Kane, D. K. Menon, D. Rueckert, and B. Glocker, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation," Medical Image Analysis, vol. 36, pp. 61–78, 2017.

[10] A. Ghodsi, J. Pan, and M. Wang, "Data augmentation approaches for improving cardiovascular risk prediction models," in Proc. IEEE Int. Conf. Bioinformatics and Biomedicine (BIBM), pp. 1072–1079, 2020.

---

## Acknowledgments

We sincerely thank Dr. Gao for her consistent guidance, expert advice, and valuable support throughout the development of this project. Her insights were instrumental in shaping the methodology and ensuring the rigor and clinical relevance of the work.

---

## Appendix A: Project Structure

```
project/
├── app.py                          # Main Flask application
├── config/                         # Configuration management
├── data/                           # Cardiovascular datasets (855K+ samples)
├── models/                         # Trained machine learning models
├── scripts/                        # Model training and analysis scripts
├── static/                         # Web application assets
├── templates/                      # HTML templates for web interface
├── utils/                          # Utility functions and testing
└── charts/                         # Performance analysis and presentation charts
```

## Appendix B: Model Performance Summary

| Metric | Value | Clinical Significance |
|--------|-------|---------------------|
| Accuracy | 90.29% | High reliability for clinical use |
| Precision | 91.2% | Low false positive rate |
| Recall | 89.8% | Good detection of high-risk patients |
| F1-Score | 90.5% | Balanced precision and recall |
| ROC AUC | 94.3% | Excellent discrimination ability |

## Appendix C: Deployment Instructions

**Local Development:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**Production Deployment:**
```bash
chmod +x deploy_railway.sh
./deploy_railway.sh
```

---

**Contact Information:**  
Sarah Akhtar  
Department of Computer Science  
University of the Pacific  
Stockton, CA, USA  
Email: s.akhtar@pacific.edu

**Project Repository:** https://github.com/s-akhtar-dev/cvd-predictor  
**Live Application:** https://cvd-predictor-production-f8be.up.railway.app/  
**License:** Educational project for academic and research purposes
