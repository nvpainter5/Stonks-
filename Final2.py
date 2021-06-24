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
from crypto_coin import Coin
import data_plot

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # bokeh sample data
    os.system("bokeh sampledata")
    # install python package
    os.system(
        'pip3 install --global-option=build_ext --global-option="-L/home/appuser/lib/" --global-option="-I/home/appuser/include/" ta-lib'
    )
    # back to the cwd
    os.chdir(default_cwd)
    print(os.getcwd())
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0")
# import library
import talib as ta

st.title("Stonks!")

st.write('---')

st.markdown('''
            The web application provides information metrics and forecast on the select stock & analyzes performance. It also provides an option to view a dashboard for Cryptocurrencies
            available on Coinbase, and gives insights on each coin.
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

    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    f"""### News for **{selected_company}**"""
    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{selected_company}.json")
    datar = r.json()
    for message in datar['messages']:
        f"#### UserName : {message['user']['username']}"
        f"#### Created At : {message['created_at']}"
        f"{message['body']}"


#---------------------------------------------------------------------#
def crypto():

    coin_col, currency_col = st.beta_columns(2)
    side = st.sidebar
    coins = coin_col.multiselect('Select the cryptos you want to watch', ['1INCH',
'AAVE', 'ADA', 'ALGO', 'ANKR', 'ATOM', 'BAL', 'BAND', 'BAT', 'BCH', 'BNT', 'BSV', 'BTC', 'Celo', 'CGLD', 'COMP', 'CRV', 'CTSI', 'CVC', 'DAI', 'DASH', 'DNT', 'ENJ', 'DOGE',
'EOS', 'ETC', 'ETH', 'FORTH', 'FIL', 'GRT', 'GNT', 'ICP', 'KNC', 'LINK', 'LOOM', 'LRC', 'LTC', 'MANA', 'MATIC', 'MIR', 'MKR', 'NMR', 'NKN', 'NU', 'OGN', 'OMG', 'OXT',
'REN', 'REP', 'RLC', 'SUSHI', 'SKL', 'SNX', 'STORJ', 'TRB', 'USDC', 'USDT', 'UMA', 'UNI', 'WBTC', 'XLM', 'XRP', 'XTZ', 'YFI', 'ZEC', 'ZRX',])
    
    currency = currency_col.selectbox('Choose a currency', ['USD', 'EUR', 'GBP'])


    date_picker = st.sidebar.date_input('Choose Date Range', [dt.date.today() - dt.timedelta(days=30), dt.date.today() + dt.timedelta(days=1)], min_value=dt.date.today() - dt.timedelta(days=365), max_value=dt.date.today() + dt.timedelta(days=1))


    date_list = []
    increment_date = date_picker[1]
    while increment_date != date_picker[0]:
        increment_date -= dt.timedelta(days=1)
        date_list.append(increment_date)


    # format_string = coin_symbol + '-' + currency
    # append the currency to each coin in the list
    for i in range(len(coins)):
        coins[i] = coins[i] + '-' + currency


    # populate a coin list
    coin_list = []
    for coin in coins:
        new_coin = Coin(coin, date_picker[0], date_picker[1])
        coin_list.append(new_coin)


    display_data = {}
    rename = {}
    if len(coin_list) != 0:
        k = 0
        for coin in coin_list:
            # set up key and assign empty list in dictionary
            key = coin.get_coin_name()
            display_data.setdefault(key, [])

            rename[0] = key

            history = data_plot.get_historic_info(coin.get_coin_name(), date_picker[0], date_picker[1], 86400)
            data_frame_of_history = pd.DataFrame(history)
            fig = go.FigureWidget(data=[go.Candlestick(x=date_list,
                    low=data_frame_of_history[1],
                    high=data_frame_of_history[2],
                    open=data_frame_of_history[3],
                    close=data_frame_of_history[4])])
            
            fig.update_layout(
                title=coin.get_coin_name() +' stock price',
                yaxis_title=coin.get_coin_name() +' price',
                xaxis_title='Date'
            )

            display_data[key].append(fig)

            daily_stats = data_plot.twent_four_hr_info(coin.get_coin_name())
            data_frame_of_stats = pd.DataFrame(daily_stats, index=[0])
            data_frame_of_stats = data_frame_of_stats.T.rename(rename, axis='columns')
            display_data[key].append(data_frame_of_stats) 

            order_book = data_plot.order_book_info(coin.get_coin_name())
            data_frame_of_order_book = pd.DataFrame(order_book, index=[0])
            data_frame_of_order_book = data_frame_of_order_book.T.rename(rename, axis='columns')
            display_data[key].append(data_frame_of_order_book)

            ticker = data_plot.ticker_info(coin.get_coin_name())
            data_frame_of_ticker = pd.DataFrame(ticker, index=[0])
            data_frame_of_ticker = data_frame_of_ticker.T.rename(rename, axis='columns')
            display_data[key].append(data_frame_of_ticker)

            k += 1
        
        for key in display_data:
            st.write("## " + key)
            st.plotly_chart(display_data[key][0], use_container_width=True)

            #coin_info = st.beta_expander("More info for - " + key)
            st.write("More info for - " + key)
            st.write("### 24hr Stats:")
            st.write(display_data[key][1])

            st.write("### Order Book:")
            st.write(display_data[key][2])

            st.write("### Ticker Info:")
            st.write(display_data[key][3])

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

    crypto()
