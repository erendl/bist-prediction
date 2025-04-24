import pandas as pd
from plotly.graph_objs.bar import selected
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import os
from functions import *
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from datetime import timedelta




st.set_page_config(
    page_title="Stocks Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide")
#Custom CSS
with open("styles.css", "r") as custom_css:
    css = custom_css.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.title("Bist Prediction")
st.write("Bist prediction is a project that aims to predict the future price of bist stocks.")
st.divider()

st.header("Bist 100 Index")
bist100 = yf.Ticker("XU100.IS")
bist100_data = bist100.history(period="6mo" , interval="1d")
bist100_dataframe = pd.DataFrame(bist100_data)


bist100_data['Daily_Change_Pct'] = bist100_data['Close'].pct_change() * 100
latest_change = bist100_data['Daily_Change_Pct'].iloc[-1]


if latest_change > 0:
    change_color = "#008000"
    arrow = "↑"
elif latest_change < 0:
    change_color = "#FF0000"
    arrow = "↓"
else:
    change_color = "#808080"
    arrow = "→"

selected_stock = "XU100"  
bist_cleangraph = yfdata_clean(selected_stock)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Closing Price:", 
              value=f"{bist100_data['Close'].iloc[-1]:.2f}{"₺"}",
              delta=f"{latest_change:.2f}% ({arrow})")

with col2:
    plotly_graph(selected_stock, bist_cleangraph)


st.divider()

#########
conn = sqlite3.connect('symbols_bist.db')
symbols_df = pd.read_sql('SELECT symbol, title FROM stocks', conn)
conn.close()

period_def_index = "2 Years"

col1, col2, col3 = st.columns(3)
with col1:
    selected_stock_title = st.selectbox(
        'Select a stock:',
        symbols_df['title'],)
    
with col2:
    selected_period_title = st.selectbox(
        'Select a period:',
        periods['title'],
        index=6)
        
with col3:
    selected_interval_title = st.selectbox(
        'Select an interval:',
        intervals['title'],
        index=2)

col1, col2= st.columns(2)
with col1:  
    new_title = st.text_input(f"Add Custom Stock (Must be listed on BIST)", placeholder="Enter the stock title (e.g. Turkish Airlines)", key="new_title")

with col2:
    new_symbol = st.text_input("    ", placeholder="Enter the stock symbol (e.g. THYAO)", key="new_symbol")

if st.button("Add Stock", key="stockButton"):
    if new_symbol and new_title and new_symbol == new_symbol.upper():
        conn = sqlite3.connect('symbols_bist.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO stocks (symbol, title) VALUES (?, ?)", (new_symbol, new_title))
        conn.commit()
        conn.close()
        st.success(f"Stock '{new_title}' added. Please Wait...")
        time.sleep(2)
        st.rerun()
    else:
        st.warning("Please enter both, stock title and the symbol. Symbol should be in uppercase.")



selected_period = get_symbols(periods, selected_period_title)
selected_interval = get_symbols(intervals, selected_interval_title)
selected_stock = get_symbolsdb(symbols_df, selected_stock_title)


df_cleandata = yfdata_clean(selected_stock, period=selected_period, interval=selected_interval)

try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("  ")
        st.subheader("  ")
        df = df_cleandata.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        df_lagged = create_lag(df, n_lags=5)
        X = df_lagged.drop(columns='Close')
        y = df_lagged['Close']

        X_train, X_test = X[:-5], X[-5:]
        y_train, y_test = y[:-5], y[-5:]

        model = XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=3, subsample=1.0, colsample_bytree=1.0)
        model.fit(X_train, y_train)
        last_known_lags = X.iloc[[-1]] 
        next_interval_pred_xgb = model.predict(last_known_lags)[0]

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader("XGBoost Model")
        st.write(f"Predicted Closing Price: {y_pred[0]:.2f} ₺")
        st.write(f"Actual Closing Price: {y_test.values[0]:.2f} ₺")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"R2 Score: {r2:.4f}")
        st.write(f"Predicted Next Interval Closing Price: {next_interval_pred_xgb:.2f} ₺")

        next_date = get_next_datetime(df_lagged.index[-1], selected_interval)
        
    with col2:
        st.subheader("  ")
        st.subheader("  ")
        st.subheader("SARIMA Model")
        df = df_cleandata.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)


        data_arima = df['Close']
        data_arima = df['Close']
        train, test = data_arima[:-5], data_arima[-5:]
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))  # burayı grid search ile optimize edebilirsin
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=len(test))
        forecast_values = forecast.predicted_mean
        mse = mean_squared_error(test, forecast_values)
        r2 = r2_score(test, forecast_values)
        next_forecast = model_fit.get_forecast(steps=len(test) + 1)
        next_interval_pred_arima = next_forecast.predicted_mean.iloc[-1]
        st.write(f"Predicted Closing Price: {forecast_values.iloc[0]:.2f} ₺")
        st.write(f"Actual Closing Price: {test.iloc[0]:.2f} ₺")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"R2 Score: {r2:.4f}")
        st.write(f"Predicted Next Interval Closing Price: {next_interval_pred_arima:.2f} ₺")        
        

        #grid_search_st(X_train, y_train) shows best params on st
        
    with col3:
        ticker = yf.Ticker(selected_stock + ".IS")
        hist = ticker.history(period="6mo") 
        
        if hist.empty:
            st.warning("Cannot find historical data for this stock.",)
        else:
            plotly_graph(selected_stock, df_cleandata, predicted_date=next_date, predicted_price=next_interval_pred_xgb, predicted_price_arima=next_interval_pred_arima)



    st.caption("⚠ Note: The actual and predicted closing prices are from 5 intervals ago. The next interval prediction shows the expected price for the upcoming interval. ")

except Exception as e:
    st.warning("Cannot find data for this stock or Period and Interval difference is too much/less.")

st.link_button("GitHub", "https://github.com/erendl")










        

    



