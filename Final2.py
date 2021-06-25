from matplotlib import colors
import streamlit as st
import datetime as dt
from datetime import date
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objs as goo
from plotly.subplots import make_subplots
import pandas as pd
from pandas_datareader import data as wb
import os
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time
import altair as alt
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot


st.title("Stonks!")

st.write('---')

st.markdown('''
            The web application provides information metrics and forecast on the select stock & analyzes performance.
            ''')



# -----------------------------------------------------------------------

def snp_500():

    # -----------------------------------------------------------------------

    # Defining start & end dates
    # start='2021-01-01'
    start = dt.datetime.today()-dt.timedelta(5 * 365) #2 years ago
    end=date.today()

    # Web Scrapping S&P 500 Companies from wikipedia
    
    df = pd.read_csv('stocks.csv')
    #st.write(df.head())
    SNP_Ticker = list(df.Symbol)

    # -----------------------------------------------------------------------

    #Streamlit: Selecting one ETF at a time
    selected_company = st.selectbox("Choose a Tickr from the list or type: ",SNP_Ticker)

    # -----------------------------------------------------------------------


    # Caching already loaded data
    @st.cache

    # Loading data using the array above & yfinance 
    #Custom function to load data
    def load_data(SNP_Ticker):
        data = yf.download(SNP_Ticker, start, end)
        data.reset_index(inplace=True)
        return data

    # -----------------------------------------------------------------------

    # Loading RAW Data frame  
    data_load_state = st.text("Data Loading...")
    data = load_data(selected_company)
    data_load_state.text("Data load completed successfully!")

    # -----------------------------------------------------------------------

    # Drawing a line
    st.write('---')

    # -----------------------------------------------------------------------

    Company_Data = yf.Ticker(selected_company) 
    Compamy_Df = Company_Data.info 

    #st.subheader('Company Profile')
    
    st.subheader(Compamy_Df['longName'])
    #st.markdown('* Company Name: ' + Compamy_Df['longName'])
    #st.markdown('* Exchange Name: ' + str(Compamy_Df['fullExchangeName']))
    st.markdown('* Market Capitalization: ' + str(Compamy_Df['marketCap']))
    #st.markdown('* Current Share Price: ' + str(Compamy_Df['postMarketPrice']))

    # -----------------------------------------------------------------------

    # Drawing a line
    st.write('---')

    # -----------------------------------------------------------------------

    #st.subheader(Compamy_Df['longName'] + ' Stock Data')
    #st.write(data.head(5))

    # -----------------------------------------------------------------------

    # Drawing a line
    #st.write('---')

    # -----------------------------------------------------------------------
    st.subheader('Stock Price Trend')
    # -----------------------------------------------------------------------

    # Closing Price
    def plot_close_price():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close']))
        fig.update_layout(
        title={
                    'text': "Stock Price Over Past Three Years",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
            })
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_close_price()
    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    st.subheader('Technical Indicators')
    # -----------------------------------------------------------------------

    # MACD
    def plot_macd():
        fig = go.Figure()
        st.write('**Stock Moving Average Convergence Divergence (MACD)**')
        st.markdown(" It can be used as a generic oscillator for any univariate series, not only price. Typically MACD is set as the difference between the 12-period simple moving average (SMA) and 26-period simple moving average (MACD = 12-period SMA − 26-period SMA), or “fast SMA — slow SMA”.The MACD has a positive value whenever the 12-period SMA is above the 26-period SMA and a negative value when the 12-period SMA is below the 26-period SMA. The more distant the MACD is above or below its baseline indicates that the distance between the two SMAs is growing.Why are the 12-period SMA called the “fast SMA” and the 26-period SMA the “slow SMA”? This is because the 12-period SMA reacts faster to the more recent price changes, than the 26-period SMA.")        
        selt = data['Close']
        exp1 = selt.ewm(span=12, adjust=False).mean()
        exp2 = selt.ewm(span=26, adjust=False).mean()
        exp3 = selt.ewm(span=9, adjust=False).mean()
        macd = exp1 - exp2
        macd.plot(label = selected_company, color = 'g')
        ax = exp3.plot(label='Signal Line', color='r')
        pl = selt.plot(ax=ax, secondary_y=True, label=selected_company)
     
        fig.add_trace(go.Scatter(x=data['Date'], y=macd))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, Color = 'green')
    
    plot_macd()


    # -----------------------------------------------------------------------

    # RSI
    def plot_rsi():
        fig = go.Figure()
        st.write('**Stock Relative Strength Index (RSI)**')
        st.markdown("It measures the magnitude of recent price changes to evaluate overbought or oversold conditions. It is displayed as an oscillator and can have a reading from 0 to 100. The general rules are: RSI >= 70: a security is overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. RSI <= 30: an oversold or undervalued condition.")
        
    
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down

        rsi = 100 - (100/(1 + rs))

        fig.add_trace(go.Scatter(x=data['Date'], y=rsi))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_rsi()
    
    # -----------------------------------------------------------------------

    #  Stock Growth (%)
    def plot_sg():
        fig = go.Figure()
        st.write('**Stock Growth (%)**')
        pct_chng= (data['Close']/ data['Close'].iloc[0] * 100)
        fig.add_trace(go.Scatter(x=data['Date'], y=pct_chng))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_sg()
    # -----------------------------------------------------------------------

    #  Daily Returns
    def plot_dr():
        fig = go.Figure()
        st.write('**Daily Returns (%)**')
        dly_rtr = round((data['Close']/ data['Close'].shift(1)) - 1,4)*100
        fig.add_trace(go.Scatter(x=data['Date'], y=dly_rtr))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_dr()
    # -----------------------------------------------------------------------

    # Volatality
    def plot_vol():
        fig = go.Figure()
        st.write('**Volatality (%)**')
        pct_chng_vol = (data['Close']).pct_change()*100
        pct_chng_vol.dropna(inplace = True, axis = 0)
        vol = pct_chng_vol.rolling(7).std()*np.sqrt(7)
        fig.add_trace(go.Scatter(x=data['Date'], y=vol))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_vol()
    # -----------------------------------------------------------------------

    # Forecast
    st.subheader('Forecast for ' + selected_company)
   
    start = '2015-01-01'
    today = date.today().strftime("%Y-%m-%d")
    n_years = st.slider("Move slider to select number of years to predict:", 1, 5)

    period = n_years*365

    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name = 'Stock Open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'Stock Close'))
        fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()
    
    df_train = data[['Date', 'Close']]

    df_train = df_train.rename(columns ={"Date": "ds", "Close": "y" })

    m = Prophet()

    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    
    f"""### News for **{selected_company}**"""
    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{selected_company}.json")
    datar = r.json()
    for message in datar['messages']:
        f"#### UserName : {message['user']['username']}"
        f"#### Created At : {message['created_at']}"
        f"{message['body']}"


#---------------------------------------------------------------------#
    st.write('---')
def placeholder():

    st.write('---')    

option = st.sidebar.selectbox(
    "Which asset would you like to choose?",
    ["Select","Stocks", "Cryptocurrency"],
)

if option == "Select":

    placeholder()

elif option == "Stocks":
    snp_500()

elif option == "Cryptocurrency":

    st.title("Coming Soon!")
