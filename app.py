import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from PIL import Image
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
from matplotlib.pyplot import rc
import matplotlib.pyplot as plt
import plotly.graph_objs as go

yf.pdr_override()

st.set_page_config(
    page_title="Stock Comparison",
    initial_sidebar_state="collapsed",
    page_icon=":dollar:",
    layout="wide"
)

@st.cache(allow_output_mutation=True)
def load_data(ticker, start, end):
    return pdr.get_data_yahoo(ticker, start=start, end=end, interval="1d")
    
def set_pub():
    rc("font", weight="bold")  # bold fonts are easier to see
    rc("grid", c="0.5", ls="-", lw=0.5)
    rc("figure", figsize=(10, 8))
    plt.style.use("seaborn-whitegrid")
    rc("lines", linewidth=1.3, color="b")


def plotData(ticker,start,end,day1,day2):
    df_stockdata = load_data(ticker, start, end)["Adj Close"]
    df_stockdata.index = pd.to_datetime(df_stockdata.index)

    set_pub()
    fig, ax = plt.subplots(2, 1)

    ax[0].set_title("Adj Close Price %s" % ticker, fontdict={"fontsize": 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values, "g-", linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].grid(True)

    ma1 = df_stockdata.rolling(days1).mean()
    ax[0].plot(ma1, "b-", label="MA %s days" % days1)
    ax[0].legend(loc="best")

    ma2 = df_stockdata.rolling(days2).mean()
    ax[0].plot(ma2, color="magenta", label="MA %s days" % days2)
    ax[0].legend(loc="best")

    ax[1].set_title("Daily Total Returns %s" % ticker, fontdict={"fontsize": 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:], "r-")
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    plt.tight_layout()
    ax[1].grid(True)
    return plt

def display_candlestick(ticker,start,end):
    data_candlestick = load_data(ticker, start, end)
    data_candlestick.reset_index(level=0, inplace=True)
    fig = go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=data_candlestick['Date'].dt.date,
                        open=data_candlestick['Open'],
                        high=data_candlestick['High'],
                        low=data_candlestick['Low'],
                        close=data_candlestick['Close'],
                        name=ticker)])
    fig.update_xaxes(type='category')
    fig.update_layout(height=500)
    return fig


st.header(f"**Welcome...!**\n")

compare_stocks = st.form("Compare Stocks")

sp500_list = pd.read_csv("SP500_list.csv")

ticker = compare_stocks.multiselect(
        "Select multiple the ticker if present in the S&P 500 index (Currently max 3 ticker supported!)",
        sp500_list["Symbol"] + " (" + sp500_list["Name"] + ")",
    )

months = compare_stocks.slider("Choose Date Range", 1, 24, 3, 1, format="%d months")

days1 = compare_stocks.slider("Business Days to roll Moving average", 5, 120, 10)
days2 = compare_stocks.slider("Business Days to roll Moving average 2", 5, 120, 20)

enable_candlestick = compare_stocks.checkbox("Enable Candlestick Chart")
enable_stocktwits = compare_stocks.checkbox("Enable Recent Stocktwits discussions")

end_date = datetime.today()
end_date_string = end_date.strftime("%Y-%m-%d")
start_date = end_date + relativedelta(months=-int(months))
start_date_string = start_date.strftime("%Y-%m-%d")

submitted = compare_stocks.form_submit_button('Submit')


if submitted:
    if len(ticker) > 3 or len(ticker) <= 0:
        st.error("You have to select tickers between 1 to 4")
    clean_ticker = [i.split(' (')[0].upper() for i in ticker ]
    cols = st.beta_columns(len(clean_ticker))
    for i, col in enumerate(cols):
        with col:
            plt = plotData(clean_ticker[i],start_date_string,end_date_string,days1,days2)
            st.pyplot(plt,use_column_width=True,use_container_width=True)
            if enable_candlestick:
                fig = display_candlestick(clean_ticker[i],start_date_string,end_date_string)
                st.plotly_chart(fig, use_container_width=True,use_column_width=True)

            if enable_stocktwits:
                f"""### Stocktwits for **{clean_ticker[i]}**"""
                r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{clean_ticker[i]}.json")
                data = r.json()
                for message in data['messages']:
                    f"#### UserName : {message['user']['username']}"
                    f"#### Created At : {message['created_at']}"
                    f"{message['body']}"
