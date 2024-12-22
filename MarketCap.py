import yfinance as yf

# Define the ticker symbol
ticker = 'XOM'

# Get the ticker object
stock = yf.Ticker(ticker)

# Get basic information
info = stock.info

print(info)