# %%
# https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
# https://towardsdatascience.com/arima-forecasting-in-python-90d36c2246d3
# %%
import streamlit as st
from datetime import date
from datetime import datetime, timedelta
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
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
import numpy as np

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
    
    # Initialize session state for ticker selection
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = ticker_list[0]
    
    # Ticker selection dropdown
    selected_ticker = st.selectbox(
        "Select Ticker",
        options=ticker_list,
        index=ticker_list.index(st.session_state.selected_ticker),
        key='ticker_selector'
    )
    st.session_state.selected_ticker = selected_ticker
    
    # Load data for all tickers
    all_data = {}
    for ticker in ticker_list:
        all_data[ticker] = mover.load_data(ticker)
    
    data = all_data[selected_ticker].copy()
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
    
    def tripleGraph(data, ticker_name):
        chart1 = data
        chart2 = data.rolling(window = 12).mean().dropna()
        chart3 = data.rolling(window = 12).std().dropna()

        df = pd.concat([chart1, chart2, chart3], axis=1)
        df.columns=['Close', 'Rolling Mean', 'Rolling Std']

        st.title(f'Rolling value decomposition on {ticker_name} stock')
        st.line_chart(df, width=800)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.line_chart(df['2018-04-28':'2019-04-28'], width=200)
        with col2:
            st.line_chart(df['2019-04-28':'2020-04-28'], width=200)
        with col3:
            st.line_chart(df['2020-04-28':'2021-04-28'], width=200)

    tripleGraph(thin_data['Close'], selected_ticker)
    # st.altair_chart([thin_data['Close'] | thin_data['Close'].rolling(window = 12).mean().dropna()])
    # %%
    # Create volume and daily returns analysis chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Trading Volume Over Time',
            'Daily Returns Distribution',
            'Price Change vs Volume',
            'Rolling Volatility (30-day)'
        ),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )
    
    # Calculate daily returns
    daily_data = data.copy()
    daily_data['Returns'] = daily_data['Close'].pct_change() * 100
    daily_data['Price_Change'] = daily_data['Close'].diff()
    daily_data['Volatility'] = daily_data['Returns'].rolling(window=30).std()
    daily_data = daily_data.dropna()
    
    # 1. Volume over time (bar chart)
    fig.add_trace(
        go.Bar(
            x=thin_data.index,
            y=thin_data['Volume'],
            marker_color='#42a5f5',
            name='Volume'
        ),
        row=1, col=1
    )
    
    # 2. Daily returns histogram
    fig.add_trace(
        go.Histogram(
            x=daily_data['Returns'],
            nbinsx=50,
            marker_color='#ab47bc',
            name='Returns %'
        ),
        row=1, col=2
    )
    
    # 3. Price change vs Volume scatter
    fig.add_trace(
        go.Scatter(
            x=daily_data['Volume'],
            y=daily_data['Returns'],
            mode='markers',
            marker=dict(
                color=daily_data['Returns'],
                colorscale='RdYlGn',
                size=5,
                opacity=0.6
            ),
            name='Price vs Vol'
        ),
        row=2, col=1
    )
    
    # 4. Rolling volatility
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data['Volatility'],
            mode='lines',
            line=dict(color='#ffa726', width=2),
            name='Volatility'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'{selected_ticker} Trading Activity & Volatility Analysis',
        height=700,
        showlegend=False,
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Return %', row=1, col=2)
    fig.update_xaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=2)
    fig.update_yaxes(title_text='Volume', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    fig.update_yaxes(title_text='Return %', row=2, col=1)
    fig.update_yaxes(title_text='Volatility %', row=2, col=2)

    st.plotly_chart(fig, key="volume-analysis-chart", use_container_width=True)
    
    # Display decomp_plot after the candlestick chart
    st.pyplot(decomp_fig)
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
    # Time Series Forecasting with SARIMAX (from timeseries_analysis.ipynb)
    st.header(f"📈 Time Series Forecasting Analysis - {selected_ticker}")
    
    # Prepare data for SARIMAX model
    y_series = thin_data['Close'].dropna()
    
    # Fit SARIMAX model with optimal parameters (similar to notebook's approach)
    try:
        mod = sm.tsa.statespace.SARIMAX(
            y_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 6),  # 6 periods for biweekly data
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = mod.fit(disp=False)
        
        # 1. ARIMA Model Diagnostics Plot
        st.subheader("🔍 SARIMAX Model Diagnostics")
        st.markdown("""
        Model diagnostics help validate our forecasting model. The plots below show:
        - **Standardized Residuals**: Should show no patterns
        - **Histogram**: Residuals should be normally distributed  
        - **Q-Q Plot**: Points should follow the diagonal line
        - **Correlogram**: Autocorrelation should be near zero
        """)
        
        diag_fig = results.plot_diagnostics(figsize=(16, 10))
        plt.tight_layout()
        st.pyplot(diag_fig)
        
        # 2. Forecast Validation - One-step ahead predictions with confidence intervals
        st.subheader("🎯 Forecast Validation")
        
        # Get predictions for the last portion of data
        split_point = len(y_series) - 20 if len(y_series) > 20 else len(y_series) // 2
        pred_start = y_series.index[split_point]
        
        pred = results.get_prediction(start=pred_start, dynamic=False)
        pred_ci = pred.conf_int()
        
        # Create validation plot
        fig_validation, ax_validation = plt.subplots(figsize=(17, 8))
        
        # Plot observed data
        y_series.plot(ax=ax_validation, label='Observed', linewidth=2)
        
        # Plot one-step ahead forecast
        pred.predicted_mean.plot(ax=ax_validation, label='One-step ahead Forecast', 
                                  alpha=0.8, linewidth=2, color='#ff7f0e')
        
        # Add confidence intervals
        ax_validation.fill_between(
            pred_ci.index,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1], 
            color='orange', 
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        ax_validation.set_xlabel('Date', fontsize=12)
        ax_validation.set_ylabel('Stock Price ($)', fontsize=12)
        ax_validation.set_title(f'{selected_ticker} - Forecast Validation', fontsize=14, fontweight='bold')
        ax_validation.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_validation)
        
        # 3. Calculate and display forecast accuracy metrics
        y_forecasted = pred.predicted_mean
        y_truth = y_series.loc[pred_start:]
        
        # Align the series for comparison
        common_idx = y_forecasted.index.intersection(y_truth.index)
        if len(common_idx) > 0:
            y_forecasted_aligned = y_forecasted.loc[common_idx]
            y_truth_aligned = y_truth.loc[common_idx]
            
            mse = ((y_forecasted_aligned - y_truth_aligned) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = np.abs(y_forecasted_aligned - y_truth_aligned).mean()
            mape = (np.abs((y_truth_aligned - y_forecasted_aligned) / y_truth_aligned) * 100).mean()
            
            # Display metrics in columns
            st.subheader("📊 Forecast Accuracy Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Squared Error", f"{mse:,.2f}")
            with col2:
                st.metric("Root MSE", f"${rmse:,.2f}")
            with col3:
                st.metric("Mean Absolute Error", f"${mae:,.2f}")
            with col4:
                st.metric("MAPE", f"{mape:.2f}%")
        
        # 4. Future Forecast with Uncertainty Bounds
        st.subheader("🔮 Future Price Forecast")
        
        # Forecast next 20 periods (approximately 40 weeks with biweekly data)
        forecast_steps = 20
        pred_future = results.get_forecast(steps=forecast_steps)
        pred_future_ci = pred_future.conf_int()
        
        fig_forecast, ax_forecast = plt.subplots(figsize=(17, 8))
        
        # Plot historical data
        y_series.plot(ax=ax_forecast, label='Historical', linewidth=2)
        
        # Plot forecast
        pred_future.predicted_mean.plot(ax=ax_forecast, label='Forecast', 
                                         linewidth=2, color='#2ca02c')
        
        # Add uncertainty bounds
        ax_forecast.fill_between(
            pred_future_ci.index,
            pred_future_ci.iloc[:, 0],
            pred_future_ci.iloc[:, 1], 
            color='green', 
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        ax_forecast.set_xlabel('Date', fontsize=12)
        ax_forecast.set_ylabel('Stock Price ($)', fontsize=12)
        ax_forecast.set_title(f'{selected_ticker} - Future Price Forecast (Next {forecast_steps} Periods)', 
                              fontsize=14, fontweight='bold')
        ax_forecast.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_forecast)
        
        # Add model summary in expander
        with st.expander("📋 View SARIMAX Model Summary"):
            st.text(str(results.summary()))
            
    except Exception as e:
        st.warning(f"Could not fit SARIMAX model: {str(e)}")
        st.info("Try selecting a different ticker or time period.")

# %%
if __name__ == '__main__':
    main()    
        