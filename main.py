# %%
# https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
# https://towardsdatascience.com/arima-forecasting-in-python-90d36c2246d3
# %%
import streamlit as st
from datetime import date
from datetime import datetime, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datamover import DataMover
from datamover import ModelPlot
import statsmodels.api as sm
from pylab import rcParams
# from statsmodels.tsa.arima_model import ARIMA  # Depricated
from statsmodels.tsa.arima.model import ARIMA
import mplfinance as mpf

# %%
def main():
    # %%
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rc('patch', force_edgecolor=True,edgecolor='black')
    plt.rc('hist', bins='auto')
    sns.set_context('notebook')
    sns.set_palette('gist_heat')
    st.set_page_config(layout='wide')
    # %%
    today = date.today().strftime("%Y-%m-%d")
    mover = DataMover("2018-04-26", today)
    ticker_list = ["AAPL", "GOOG", "MSFT", "TSLA"]
    data = mover.load_data(ticker_list[0])
    # %%
    def datetime_range(start=None, end=None):
        span = end - start
        for i in range(span.days + 1):
            yield start + timedelta(days=i)
    date_index = list(datetime_range(start=datetime(2018, 4, 26), end=datetime(2021, 4, 26)))
    # %%
    data.Date = pd.to_datetime(data.Date)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True, ascending=True)
    # %%
    thin_data = data.resample('2w').mean()
    # %%
    mplot = ModelPlot()
    decomp_fig = mplot.decomp_plot(thin_data['Close'])
    st.pyplot(decomp_fig)
    
    def tripleGraph(data):
        chart1 = data
        chart2 = data.rolling(window = 12).mean().dropna()
        chart3 = data.rolling(window = 12).std().dropna()

        df = pd.concat([chart1, chart2, chart3], axis=1)
        df.columns=['Close', 'Rolling Mean', 'Rolling Std']

        st.title('Rolling value decomposition on Apple stock')
        st.line_chart(df, width=800)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.line_chart(df['2018-04-28':'2019-04-28'], width=200)
        with col2:
            st.line_chart(df['2019-04-28':'2020-04-28'], width=200)
        with col3:
            st.line_chart(df['2020-04-28':'2021-04-28'], width=200)

    tripleGraph(thin_data['Close'])
    # st.altair_chart([thin_data['Close'] | thin_data['Close'].rolling(window = 12).mean().dropna()])
    # %%
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=thin_data['Close'], x=thin_data['Close'].index, name="Close"))
    fig.add_trace(go.Scatter(y=thin_data['Open'], x=thin_data['Open'].index, name="Open"))
    fig.add_trace(go.Scatter(y=thin_data['High'], x=thin_data['High'].index, name="High"))
    fig.add_trace(go.Scatter(y=thin_data['Low'], x=thin_data['Low'].index, name="Low"))
    fig.update_layout(legend_title_text='I have a title')

    st.plotly_chart(fig, key="seaborn-darkgrid")
    # %%
    decomp = sm.tsa.seasonal_decompose(thin_data['Close'], model='additive', extrapolate_trend='freq', period=6)
    fig, axes = plt.subplots(4, 1, figsize=(17, 12))
    axes[0].plot(thin_data['Close'])
    axes[1].plot(decomp.trend)
    axes[2].plot(decomp.seasonal)
    axes[3].scatter(thin_data.index.values, decomp.resid)
    
    st.pyplot(fig)
    # %%
    fig = mplot.plot_arima(thin_data, 'Close')
    st.pyplot(fig)
# %%
if __name__ == '__main__':
    main()    
        