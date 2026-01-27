# %%
import streamlit as st
from datetime import date
import datetime
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt
import seaborn as sns
import json
# from statsmodels.tsa.arima_model import ARIMA  # Depricated
from statsmodels.tsa.arima.model import ARIMA

# %%
class DataMover:
    start = ""
    stop = ""
    
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def load_data(self, ticker):
        df = yf.download(ticker, self.start, self.stop)
        # Flatten MultiIndex columns if present (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def load_datas(self, ticker_list):
        train = []
        for stock in ticker_list:
            df = self.load_data(stock)
            date = df['Date']
            train.append(df['Close'])
        train_df  = pd.concat(train, axis=1)
        train_df.columns = ticker_list
        train_df.set_index(date, inplace=True)
        train_df.reset_index(inplace=True)
        return train_df

    def write_json(self, df, path):
        df.to_json(r'{}'.format(path))
    
    def read_json(self, path):
        with open(path, "r") as file_ptr:
            obj = json.load(file_ptr)
        return obj
    
class ModelPlot:

    def __init__(self):
        # plt.style.use('ggplot')
        plt.rc('patch', force_edgecolor=True,edgecolor='black')
        plt.rc('hist', bins='auto')
        style.use('seaborn-v0_8-darkgrid')
        sns.set_context('notebook')
        sns.set_palette('gist_heat')

    def decomp_plot(self, df):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.plot(df)
        ax.plot(df.rolling(window = 12).mean().dropna(), color='g')
        ax.plot(df.rolling(window = 12).std().dropna(), color='blue')
        ax.set_title('Rolling mean')
        ax.legend(['Apple', 'mean', 'std'])
        return fig

    def plot_raw_data(self, data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['High'], name='Apple stock'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Low'], name='Google stock'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)   

    def plot_arima(self, df, label):
        indx = df.index
        start = indx[-20]
        end = indx[-1]

        arima_model = ARIMA(df[label][:-20], order=(2,1,2)).fit()
        pred = arima_model.predict(start, end)
        
        rolling_mean = df[label].rolling(window = 12).mean().dropna()
        rolling_std = df[label].rolling(window = 12).std().dropna()

        arima_model = ARIMA(rolling_mean[:-20], order=(1,1,1), dates=df.index).fit()
        pred_mean = arima_model.predict(start, end)

        arima_model = ARIMA(rolling_std[:-20], order=(1,1,1), dates=df.index).fit()
        pred_std = arima_model.predict(start, end)
        

        fig, ax = plt.subplots(figsize=(17,8))
        ax.plot(df[label], alpha=0.5)
        ax.plot(rolling_mean, color='g', alpha=0.5)
        ax.plot(rolling_std, color='blue', alpha=0.5)

        # Predicted 
        ax.plot(pred)
        ax.plot(pred_mean, color='g')
        ax.plot(pred_std, color='b')
        ax.set_title('Rolling mean')
        ax.legend([label, 'mean', 'std', 'predicted'])
        return fig  
# %%
