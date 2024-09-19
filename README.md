# House Price Prediction

## House Price Prediction Using Machine Learning
Predicting house prices is a crucial task in real estate and financial planning. House prices depend on various factors such as the size of the house, number of rooms, location, and more. The relationship between these variables and the final price is complex and non-linear. Using Machine Learning to predict house prices can help create accurate pricing models, assisting buyers, sellers, and investors in making informed decisions. This project leverages machine learning techniques to build a predictive model that estimates house prices based on key features.

## 1. Problem Statement
Predicting house prices based on various features such as the number of rooms, house size, location, and other related factors. House price prediction is important for real estate investors, buyers, sellers, and financial institutions to make informed decisions.

Please refer to the [Data Analysis (House).ipynb](https://github.com/xiaozhu1110/House-Prices-Regression/blob/main/Data%20Analysis%20(House).ipynb) for code.

## 2. Data Description
The dataset is obtained from the Kaggle Housing Prices Competition.

  * Number of instances: 1460
  * Number of attributes: 81 (including the target variable)
### Attribute Information
#### Inputs
* Lot Area
* Overall Quality
* Year Built
* Total Basement Area
* Number of Bathrooms
* Garage Area
* Neighborhood
* Number of Bedrooms

And more (see dataset for full attribute list)
All numerical attributes are measured in appropriate units like square feet or ordinal scales, while categorical attributes represent neighborhood, material quality, etc.

## Output
Sale Price (The target variable representing the price of the house)

## 3. Modelling and Evaluation

- **Algorithms used**:
  - Linear Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - XGBoost

- **Metric**: Since the target variable is continuous (house price), the following regression evaluation metrics have been used:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score (Coefficient of Determination)
 
## 4. Results

### Model Evaluation and Performance

In this project, we trained and evaluated multiple machine learning models to predict the target variable. The models included were:
- **MLP (Multi-layer Perceptron)**
- **Linear Regression**
- **XGBoost**
- **Ensemble of Models**

We evaluated the models using **RMSE** (Root Mean Squared Error) as the performance metric. Below is a comparison of the models’ performance:

---

### 1. Model Parameters and RMSE Results

The table below summarizes the best parameters found during the grid search for each model along with their respective RMSE values:

| Model               | Best Parameters                                                                                             | RMSE    |
|---------------------|------------------------------------------------------------------------------------------------------------|---------|
| **MLP**             | `activation: relu`, `alpha: 0.5`, `hidden_layer_sizes: (10,)`, `learning_rate: adaptive`, `solver: sgd`     | 0.1403  |
| **Linear Regression**| Default parameters                                                                                         | 0.1424  |
| **XGBoost**         | `learning_rate: 0.01`, `max_depth: 3`, `n_estimators: 500`                                                  | 0.1340  |
| **Ensemble Model**  | N/A (Stacking ensemble of MLP, Linear Regression, XGBoost)                                                  | 0.1378  |

---

### 2. Performance Comparison

The chart below visualizes the performance of each model in terms of RMSE:

![Model RMSE Comparison](https://github.com/xiaozhu1110/House-Prices-Regression/blob/main/Model%20RMSE%20Comparison.png)

#### Interpretation:
- **XGBoost** achieved the best performance with the lowest RMSE of **0.1340**, indicating that it is the most accurate individual model for this problem.
- The **Ensemble Model**, which combines the predictions from MLP, Linear Regression, and XGBoost, resulted in an RMSE of **0.1378**, showing that stacking improved performance over the average of the individual models.
- The **MLP** model performed decently with an RMSE of **0.1403**.
- **Linear Regression**, being the simplest model, had the highest RMSE of **0.1424**, making it the least accurate.

---

### 3. Code Snippet

If you'd like to try out the models and reproduce the results, here's a Python code snippet that trains the models using a stacked ensemble:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define models
estimators = [
    ('mlp', MLPRegressor(hidden_layer_sizes=(10,), activation='relu', alpha=0.5, learning_rate='adaptive', solver='sgd')),
    ('lr', LinearRegression()),
    ('xgb', XGBRegressor(learning_rate=0.01, max_depth=3, n_estimators=500))
]

# Stacking Regressor
stacked_model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())

# Split your dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the model
stacked_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = stacked_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
