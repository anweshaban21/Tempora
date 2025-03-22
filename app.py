import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
start = '2015-01-01'
today = date.today().strftime("%Y-%m-%d")
from Prophecy import forecastData
# from Prophecy import load_data
# Title of the Streamlit app
st.title('Stock Prediction App')


stocks = ('AAPL', 'GOOG', 'MSFT', 'GME','INFY.NS','ZOMATO.NS','TCS.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.selectbox("Select Years of Prediction:", [1, 2, 3, 4])
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

# Loading data from Yahoo Finance for the selected stock
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')
forecast=forecastData(data,period)

# if data.empty:
#     st.error("Failed to load stock data. Please try again later or choose another stock.")
#     st.stop()
forecast = forecast.rename(columns={
    "ds": "Date",
    "yhat": "Predicted Price",
    "yhat_lower": "Lower Bound",
    "yhat_upper": "Upper Bound"
})

forecast[["Date", "Predicted Price", "Lower Bound", "Upper Bound"]].tail()

st.write(forecast[["Date", "Predicted Price", "Lower Bound", "Upper Bound"]].tail())

fig = px.line(forecast, x="Date", y=["Predicted Price", "Lower Bound", "Upper Bound"],
              labels={"value": "Stock Price", "variable": "Legend"},
              title="Stock Price Forecast Over Time")


st.plotly_chart(fig)






