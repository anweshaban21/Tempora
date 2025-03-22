import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import date
start = '2020-01-01'
today = date.today().strftime("%Y-%m-%d")
# from app import load_data

# data = yf.download('INFY.NS', start=start, end=today)
def forecastData(data,n):
    df=pd.DataFrame(data)
    df.tail()


    df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df.head()
    m = Prophet()
    m.fit(df[["ds", "y"]])
    future = m.make_future_dataframe(periods=n)
    future.tail()
    forecast = m.predict(future)
    
    # fig1 = m.plot(forecast)
    # fig2 = m.plot_components(forecast)
    from prophet.plot import plot_plotly, plot_components_plotly
    plot_plotly(m, forecast)
    return forecast