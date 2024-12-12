import ccxt
import pandas as pd
from fpdf import FPDF
import requests
from datetime import datetime, timedelta

# Initialize Binance exchange
binance = ccxt.binance()

# Fetch USDT tickers from Binance API
def get_usdt_tickers():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        usdt_tickers = [symbol["symbol"] for symbol in data["symbols"] if symbol["symbol"].endswith("USDT") and symbol["status"] == "TRADING"]
        return usdt_tickers
    except requests.RequestException as e:
        print(f"Error fetching tickers: {e}")
        return []

# Fetch OHLCV data (Open, High, Low, Close, Volume) for a symbol
def get_crypto_data(symbol='BTC/USDT', timeframe='1d', limit=2000):
    try:
        bars = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Calculate RSI and store it in the dataframe
        df['RSI'] = calculate_rsi(df).round(2)

        # Calculate Stochastic RSI and unpack the %K and %D values
        stoch_rsi_k_smoothed, stoch_rsi_d = calculate_stochastic_rsi_crypto(df)

        # Store only the Stochastic RSI %D value in the dataframe, rounded to 2 decimal places
        df['Stochastic_RSI_D'] = stoch_rsi_d.round(2)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

# RSI calculation
def calculate_rsi(data, period=14):
    delta = data['close'].diff()  # Price differences between consecutive days
    gain = delta.where(delta > 0, 0)  # Gains (positive differences)
    loss = -delta.where(delta < 0, 0)  # Losses (negative differences)

    # Calculate RMA (Rolling Moving Average) for gains and losses
    alpha = 1 / period  # Smoothing factor
    rma_gain = gain.ewm(alpha=alpha, adjust=False).mean()  # Exponential moving average of gains
    rma_loss = loss.ewm(alpha=alpha, adjust=False).mean()  # Exponential moving average of losses

    # Calculate the Relative Strength (RS) and the RSI
    rs = rma_gain / rma_loss
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    return rsi

# Calculate Stochastic RSI
def calculate_stochastic_rsi_crypto(data, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    # Step 1: Calculate RSI
    rsi_values = calculate_rsi(data, period=length_rsi)

    # Step 2: Calculate Stochastic RSI
    rsi_min = rsi_values.rolling(window=length_stoch).min()  # Min RSI over period
    rsi_max = rsi_values.rolling(window=length_stoch).max()  # Max RSI over period
    stoch_rsi_k = 100 * (rsi_values - rsi_min) / (rsi_max - rsi_min)  # Stochastic RSI %K

    # Step 3: Smooth the %K and calculate %D
    stoch_rsi_k_smoothed = stoch_rsi_k.rolling(window=smooth_k).mean()  # Smooth %K
    stoch_rsi_d = stoch_rsi_k_smoothed.rolling(window=smooth_d).mean()  # Smooth %D (3-period SMA)

    return stoch_rsi_k_smoothed, stoch_rsi_d

# Fetch data for all cryptos and return a dictionary
def fetch_all_crypto_data(crypto_list):
    result = {}
    for symbol in crypto_list:
        print(f"Fetching data for {symbol}...")
        data = get_crypto_data(symbol=symbol)
        
        if not data.empty and len(data) >= 900:
            last_close_price = data['close'].iloc[-1]
            last_volume = data['volume'].iloc[-1]
            last_day_product = last_close_price * last_volume
            result[symbol] = data[['timestamp', 'close', 'RSI', 'Stochastic_RSI_D']].tail(1500)
            result[symbol]['symbol'] = symbol  # Add a column for the symbol
            result[symbol]['last_day_product'] = last_day_product  # Add product of close and volume

            print(f"Completed {symbol}")
    return result

# Function to get max minus one date data
def get_maximum_minus_one_date_data(crypto_data):
    max_minus_one_date_data = {}
    for symbol, data in crypto_data.items():
        # Sort the data by timestamp and get the latest date
        max_date = data['timestamp'].max()  # Find the maximum (latest) date
        
        # Filter the data to get the row for the day before the maximum date
        max_minus_one_date = max_date - pd.Timedelta(days=1)  # Calculate the date one day before the maximum date
        
        # Extract the row corresponding to max_minus_one_date
        row = data[data['timestamp'] == max_minus_one_date]
        
        if not row.empty:  # If a row for the date exists
            max_minus_one_date_data[symbol] = row
        else:
            print(f"No data found for {symbol} on {max_minus_one_date.strftime('%Y-%m-%d')}")
            max_minus_one_date_data[symbol] = None
    
    return max_minus_one_date_data

# Generate CSV reports instead of PDFs    
def generate_pdf(crypto_data, market_name, target_date):
    rsi_less_than_51_data = []
    combined_alert_data = []
    stoch_rsi_less_than_1_data = []

    for symbol, data in crypto_data.items():
        try:
            if data.empty or len(data) < 900:
                continue

            rsi_value_today = data['RSI'].iloc[-2]
            lowest_rsi_900_td = data['RSI'].min()
            stoch_rsi_d_today = data['Stochastic_RSI_D'].iloc[-2]

            # RSI less than 1.51 * lowest RSI of the last 900 days
            if rsi_value_today <= 1.51 * lowest_rsi_900_td:
                rsi_less_than_51_data.append({
                    'Crypto': symbol,
                    'RSI': rsi_value_today,
                    'Lowest_RSI_900_TD': lowest_rsi_900_td,
                    'Date': target_date
                })

            # Combined alert: RSI and Stochastic RSI both meet conditions
            if rsi_value_today <= 1.51 * lowest_rsi_900_td and stoch_rsi_d_today <= 1:
                combined_alert_data.append({
                    'Crypto': symbol,
                    'RSI': rsi_value_today,
                    'Stochastic_RSI': stoch_rsi_d_today,
                    'Date': target_date
                })

            # Stochastic RSI less than 1
            if stoch_rsi_d_today <= 1:
                stoch_rsi_less_than_1_data.append({
                    'Crypto': symbol,
                    'Stochastic_RSI': stoch_rsi_d_today,
                    'Date': target_date
                })

        except Exception as e:
            print(f"Error for {market_name} symbol {symbol}: {e}")
            continue

    # Convert lists to DataFrames
    rsi_less_than_51_df = pd.DataFrame(rsi_less_than_51_data)
    combined_alert_df = pd.DataFrame(combined_alert_data)
    stoch_rsi_less_than_1_df = pd.DataFrame(stoch_rsi_less_than_1_data)

    # Define helper function to create a PDF from DataFrame
    def df_to_pdf(df, title, pdf):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, txt=title, ln=True, align='C')

        pdf.set_font('Arial', '', 12)
        col_width = pdf.w / (len(df.columns) + 1)
        row_height = pdf.font_size

        # Add header
        for col in df.columns:
            pdf.cell(col_width, row_height * 2, col, border=1)

        pdf.ln(row_height * 2)

        # Add data
        for row in df.itertuples(index=False):
            for value in row:
                pdf.cell(col_width, row_height * 2, str(value), border=1)
            pdf.ln(row_height * 2)

    # Generate PDF files
    pdf = FPDF()

    rsi_less_than_51_file_path = f"{market_name}_RSI_Lessthan_51.pdf"
    combined_alert_file_path = f"{market_name}_Combined_Alert.pdf"
    stoch_rsi_less_than_1_file_path = f"{market_name}_StochRSI_Lessthan_1.pdf"

    df_to_pdf(rsi_less_than_51_df, f"RSI Less Than 1.51*Lowest RSI of Last 900 Days ({target_date})", pdf)
    pdf.output(rsi_less_than_51_file_path)

    pdf = FPDF()  
    df_to_pdf(combined_alert_df, f"Combined Alert (RSI and Stochastic RSI) ({target_date})", pdf)
    pdf.output(combined_alert_file_path)

    pdf = FPDF()  
    df_to_pdf(stoch_rsi_less_than_1_df, f"Stochastic RSI Less Than 1 ({target_date})", pdf)
    pdf.output(stoch_rsi_less_than_1_file_path)

    return [rsi_less_than_51_file_path, combined_alert_file_path, stoch_rsi_less_than_1_file_path]
# Main script
if __name__ == '__main__':
    crypto_tickers = get_usdt_tickers()
    crypto_data = fetch_all_crypto_data(crypto_tickers)
    max_minus_one_data = get_maximum_minus_one_date_data(crypto_data)
    #target_date = datetime.now().strftime('%Y-%m-%d')
    today = datetime.now()
    trading_day = today - timedelta(days=1)
    target_date = trading_day.strftime('%Y-%m-%d')
    # Generate PDFs for the data fetched
    generate_pdf(crypto_data, "CRYPTO", target_date)

