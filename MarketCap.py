import yfinance as yf

# Define the ticker symbol
ticker = 'AAPL'

# Get the ticker object
stock = yf.Ticker(ticker)

# Get basic information
info = stock.info['marketCap']

print(info)