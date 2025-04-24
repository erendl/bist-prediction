import pandas as pd
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from datetime import timedelta


class periods:
    def __init__(self):
        self.symbols = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']
        self.titles = [
            '1 Day', '5 Days', '1 Month', '3 Months', '6 Months',
            '1 Year', '2 Years', '5 Years', '10 Years'
        ]
        self.df = pd.DataFrame({
            'symbols': self.symbols,
            'title': self.titles
        })
periods = periods().df

class intervals:
    def __init__(self):
        self.symbols = ['60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        self.titles = [
            '60 Minutes', '90 Minutes', '1 Hour', '1 Day', '5 Days',
            '1 Week', '1 Month', '3 Months'
        ]
        self.df = pd.DataFrame({
            'symbols': self.symbols,
            'title': self.titles
        })

intervals = intervals().df




##################

def yfdata_clean(selected_stock, period="6mo", interval="1d"):
    data = yf.download(f"{selected_stock}.IS", period=period, interval=interval)
    data = data.dropna()
    data = data[['Close']]
    data_name = pd.DataFrame(data)
    data_name.to_csv('temp_stocks.csv')
    data_name = pd.read_csv("temp_stocks.csv", skiprows=2)
    data_name.columns = ['Date', 'Close']
    os.remove('temp_stocks.csv')
    return data_name


def plotly_graph(selected_stock, dataname,predicted_date=None, predicted_price=None, predicted_price_arima=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=dataname['Date'],
    y=dataname['Close'],
    mode='lines+markers',
    name=selected_stock,
    line=dict(color='royalblue', width=2)
    ))

    z = np.polyfit(np.arange(len(dataname)), dataname['Close'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=dataname['Date'],
        y=p(np.arange(len(dataname))),
        mode='lines',
        name='Trendline',
        line=dict(color='orange',width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[predicted_date],
        y=[predicted_price],
        mode='markers',
        name='XGBoost Pred',
        marker=dict(color='red', size=10, symbol='diamond'),
    ))
    fig.add_trace(go.Scatter(
        x=[predicted_date],
        y=[predicted_price_arima],
        mode='markers',
        name='Sarima Pred',
        marker=dict(color='purple', size=10, symbol='diamond'),
    ))


    fig.update_layout(
        title=f'{selected_stock} Trendline',
        xaxis_title='Date',
        yaxis_title='Close Price',
    )

    st.plotly_chart(fig)

def get_symbols(df_name, selected_title):
    return df_name.loc[df_name['title'] == selected_title, 'symbols'].values[0]

def get_symbolsdb(df, title):
    return df.loc[df['title'] == title, 'symbol'].values[0]

################

def create_lag(df, n_lags=5):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    return df

def grid_search_st(X_train, y_train):
    param_grid = { 
    'n_estimators': [400,500,600,700],
    'learning_rate': [0.02],
    'max_depth': [3],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
    }
    grid_search = GridSearchCV( 
        estimator=XGBRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return st.write(grid_search.best_params_)

def get_next_datetime(last_date, interval_symbol):
    if interval_symbol.endswith("m"): 
        minutes = int(interval_symbol.replace("m", ""))
        return last_date + timedelta(minutes=minutes)
    elif interval_symbol.endswith("h"): 
        hours = int(interval_symbol.replace("h", ""))
        return last_date + timedelta(hours=hours)
    elif interval_symbol == "1d":
        return last_date + timedelta(days=1)
    elif interval_symbol == "5d":
        return last_date + timedelta(days=5)
    elif interval_symbol == "1wk":
        return last_date + timedelta(weeks=1)
    elif interval_symbol == "1mo":
        return last_date + pd.DateOffset(months=1)
    elif interval_symbol == "3mo":
        return last_date + pd.DateOffset(months=3)
    else:
        return last_date 


    
        
