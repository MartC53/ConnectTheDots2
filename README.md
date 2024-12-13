# ConnectTheDots2
The goal of this project is to produce an ensemble model that accurately quantifies puncta formation in fluorescent images over a large dynamic range, these puncta are nucleation sites of DNA amplification. Previous work in the Posner Research group has shown that puncta counts correlate with the concentration of initial DNA concentration.

## Motivation
### Abstract
Without a cure, antiretroviral therapy (ART) is the primary tool for treating individuals living with HIV (Drain et Al. 2020). Monitoring HIV viral load is an important element of care for individuals on ART, as virally suppressed patients cannot transmit HIV to others. As a result, combating the HIV epidemic requires timely and accurate viral load testing.
Despite the importance of viral load testing, existing testing technologies present implementation challenges, particularly in low- and middle-income countries (Grieg et Al, 2020). Currently, viral load testing requires Nucleic Acid Tests (NATs), which detect nucleic acid from disease and infection. Detection is based on amplifying low levels of DNA/RNA allowing for detection of a single strand of DNA/RNA. The gold standard for quantitative nucleic acid testing is quantitative polymerase chain reaction (qPCR). However, qPCR is:
* slow
* expensive 
* fragile 

Isothermal DNA amplification technologies, like recombinase polymerase amplification (RPA) have been put forth that are faster, cheaper, and more robust than qPCR. Yet isothermal amplification technologies are limited in their diagnostic capabilities as they are qualitative. However, **Recent studies in the Posner Lab Group have shown that RPA, an isothermal NAT, can also be quantitative through a spot nucleation to initial copy correlation** [1]. Similar nucleation site analysis has been applied to other assays and targets that used ML to produce a quantification model which rivals our linear range [2]. Thus, we are interested in applying ML models to improve the linear range of our assay.
1.  Quantitative Isothermal Amplification on Paper Membranes using Amplification Nucleation Site Analysis
Benjamin P. Sullivan, Yu-Shan Chou, Andrew T. Bender, Coleman D. Martin, Zoe G. Kaputa, Hugh March, Minyung Song, Jonathan D. Posner
bioRxiv 2022.01.11.475898; doi: https://doi.org/10.1101/2022.01.11.475898 
2. Membrane-Based In-Gel Loop-Mediated Isothermal Amplification (mgLAMP) System for SARS-CoV-2 Quantification in Environmental Waters
Yanzhe Zhu, Xunyi Wu, Alan Gu, Leopold Dobelle, Clément A. Cid, Jing Li, and Michael R. Hoffmann
Environmental Science & Technology 2022 56 (2), 862-873
DOI: 10.1021/acs.est.1c04623

For more information please see [Further details in the wiki](https://github.com/MartC53/QIAML/wiki/Further-details)

## Methods
### Image Prepocessing 
Prior to image processing, raw images were opened as numerical arrays for computational analysis. A Gaussian blur with a 5×5 kernel was applied to each image to soften edges and mitigate darkshot noise. To reduce interpuncta fluorescence, a temporal moving average was implemented by subtracting the average of the previous three frames from the current frame. Background subtraction was performed by averaging the first 20 frames to establish a baseline, which was subtracted from all subsequent frames to normalize pixel intensities to zero and remove autofluorescence artifacts. Finally, a circular mask was applied to exclude edge artifacts by setting the pixel intensities along the image boundaries to zero.
### Feature extraction 
The following features were extracted from the processed images to characterize puncta dynamics and fluorescence behavior. The maximum number of puncta identified over the entire time course was recorded, along with the number of puncta detected at 7.5 minutes. The temporal dynamics of puncta coverage were assessed by calculating the area occupied by puncta over time and the bulk fluorescence intensity of the puncta over time. Additionally, the percentage of the image area occupied by puncta at 7.5 minutes was quantified. The average size of individual puncta was determined as the total puncta area divided by the number of puncta. To capture temporal changes, the maximum rate of fluorescence change over a 10-second interval and the maximum rate of area change over a 10-second interval were computed.

### Model training

Three machine learning models were trained and evaluated for their ability to predict nucleation site formation. These models included Linear Regression, Random Forest, and XGBoost. Hyperparameter tuning was performed for the Random Forest and XGBoost models using GridSearchCV, while the Linear Regression model was used without tuning. Below are the details of the models, their performance metrics, and the saved plots.

#### Linear Regression
- **Hyperparameters:** None (default Linear Regression model)
- **Average Train MSE:** 0.0090
- **Average Test MSE:** 0.0127
- **Plot:** Saved as `result/LinearRegression_predictions.png`

#### Random Forest
- **Best Hyperparameters:**
  - `max_depth`: None
  - `min_samples_leaf`: 2
  - `min_samples_split`: 2
  - `n_estimators`: 100
- **Average Train MSE:** 0.0028
- **Average Test MSE:** 0.0152
- **Plot:** Saved as `result/RandomForestRegressor_predictions.png`

#### XGBoost
- **Best Hyperparameters:**
  - `colsample_bytree`: 0.9
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `n_estimators`: 300
  - `reg_alpha`: 0
  - `reg_lambda`: 1
  - `subsample`: 0.8
- **Average Train MSE:** 0.0000
- **Average Test MSE:** 0.0138
- **Plot:** Saved as `result/XGBRegressor_predictions.png`

#### Results Summary
The model performance metrics suggest that:
- The **Linear Regression model** exhibited the lowest average test MSE (0.0127), indicating its generalizability across the dataset.
- The **Random Forest model** showed slightly higher test MSE, likely due to overfitting as reflected by its significantly lower train MSE (0.0028).
- The **XGBoost model** achieved competitive test MSE (0.0138), benefiting from its tuned hyperparameters.

#### Best Model
Based on the evaluation results, the Linear Regression model was selected as the best-performing model and saved to `result/linear_regression.pkl`.