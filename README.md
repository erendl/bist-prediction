# Bist Prediction 

Bist Prediction is a machine learning-based project that aims to forecast the future closing prices of stocks listed on Borsa İstanbul (BIST). It leverages two different time series prediction models:

- XGBoost Regressor
- SARIMA (Seasonal ARIMA)

The project is presented with a Streamlit dashboard for easy interaction and visualization.

---

## Features

- Select and visualize BIST100 index and individual stocks
- Predict next closing price using XGBoost and SARIMA
- Compare actual vs predicted values for the last 5 intervals
- Dynamic trendlines and prediction markers on interactive Plotly graphs
- Add custom BIST stocks manually
- Auto-update BIST symbols from TradingView via scraping
- Metrics including MSE and R2 score for evaluation

---

## Installation & Usage

1. Clone the repository:

   git clone https://github.com/erendl/bist-prediction.git
   cd bist-prediction

2. Install the required dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run main.py

---

## File Structure

- main.py  
  The main Streamlit application. It loads stock data, runs models, displays predictions and graphs.  
  Tip: Check line 184+ for `grid_search_st()` function to optimize XGBoost parameters for better accuracy.

- functions.py  
  Contains helper functions for:
  - Data fetching and cleaning
  - Feature engineering (lag creation)
  - Model evaluation
  - Plotting with Plotly
  - Date interval handling
  - Model parameter optimization via GridSearch

- get_symbols.py  
  Scrapes and updates BIST stock symbols from TradingView into a local SQLite database (symbols_bist.db).

- styles.css  
  Custom styling for the Streamlit interface.

- symbols_bist.db  
  SQLite database that stores stock titles and symbols.

---

## Models Used

### 1. XGBoost Regressor
- Uses lag features (previous 5 closing prices) for training.
- Efficient and accurate for short-term predictions.
- Tunable via grid_search_st().

### 2. SARIMA (Seasonal ARIMA)
- Suitable for capturing seasonal patterns and trends.
- Applied on raw time series without feature engineering.

---

## To-Do / Ideas for Improvement

- Use Prophet or LSTM for alternative modeling

---

## Resources

- Streamlit Documentation: https://docs.streamlit.io/
- Yahoo Finance API (yfinance): https://pypi.org/project/yfinance/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- SARIMA in statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

---

## Author

Developed by  
GitHub: https://github.com/erendl

---

## Disclaimer

This project is for educational and experimental purposes only. Predictions should not be used as financial advice.
