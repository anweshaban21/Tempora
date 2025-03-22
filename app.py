import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
# Define start and end dates
start = '2015-01-01'
today = date.today().strftime("%Y-%m-%d")

# Title of the Streamlit app
st.title('Stock Prediction App')

# Select stock
stocks = ('AAPL', 'GOOG', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Select years for prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Cache function for loading data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# #checking empty data
# if data.empty:
#     st.error("Failed to load stock data. Please try again later or choose another stock.")
#     st.stop()
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data.dropna(inplace=True)
# Display raw data
st.subheader('Raw data')
st.write(data.tail())



#Plotting the data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter (x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter (x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)
chart_data = data[[('Date', ''), ('Close', selected_stock)]]
chart_data.columns = ['Date', 'Close']  # Rename columns for easier use
chart_data.set_index('Date', inplace=True)
st.line_chart(chart_data)


# Prepare data for Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Train Prophet model
m = Prophet()
m.fit(df_train)

# # Create future dates & predict
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Display forecast data
# st.subheader('Forecast data')
# st.write(forecast.head())

# # Plot forecast
# st.subheader('Forecast Plot')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# # Show forecast components
# st.subheader('Forecast Components')
# fig2 = m.plot_components(forecast)
# st.write(fig2)
