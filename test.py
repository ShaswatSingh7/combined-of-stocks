import crypto
import os
import pandas as pd
import yfinance as yf
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
import ssl
from fpdf import FPDF
import ftplib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
from filelock import FileLock
import pytz
import asyncio
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed


# Constants
ftp_server = 'ftp.nasdaqtrader.com'
ftp_path = '/SymbolDirectory/nasdaqlisted.txt'
CACHE_DIR = 'cache'
CACHE_TTL = 3600 * 14  # Cache for 24 hours
saudi_tz = pytz.timezone('Asia/Riyadh')  # Saudi time zone



# SSL Fix
try:
    _create_unverified_https_context = ssl._create_unverified_context # Fixed the typo here
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Telegram Bot Configuration
bot_token = '7446600645:AAH-rqYt7qqbB3kxb58icrBgV_gN19g92Uk'  # Replace with your bot token
chat_id = '1263659789'  # Replace with your chat ID
bot = telegram.Bot(token=bot_token)


# Function to find the previous trading day for the general market (ignoring weekends)
def get_previous_trading_day():
    today = datetime.now()

    if today.weekday() == 0:  # If today is Monday, go back to Friday
        previous_trading_day = today - timedelta(days=3)
    else:
        previous_trading_day = today - timedelta(days=1)

    while previous_trading_day.weekday() >= 5:  # Ensure it's not a weekend
        previous_trading_day -= timedelta(days=1)

    return previous_trading_day

# Function to find the previous trading day for Saudi Exchange
def get_previous_trading_day_saudi():
    today = datetime.now()

    # Determine the last trading day based on current day
    if today.weekday() == 0:  # Monday, go back to Thursday
        previous_trading_day = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday, go back to Thursday
        previous_trading_day = today - timedelta(days=3)
    elif today.weekday() == 5:  # Saturday, go back to Thursday
        previous_trading_day = today - timedelta(days=2)
    elif today.weekday() == 4:  # Friday, go back to Thursday
        previous_trading_day = today - timedelta(days=1)
    else:  # For other weekdays (Tuesday, Wednesday, Thursday)
        previous_trading_day = today - timedelta(days=1)

    return previous_trading_day

# Set the target dates
target_date = get_previous_trading_day()
target_date_saudi = get_previous_trading_day_saudi()
extended_start_date = target_date - timedelta(days=1500)
extended_start_date_saudi = target_date_saudi - timedelta(days=1500)



# Ensure cache and PDF directories exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Retry mechanism with exponential backoff
@sleep_and_retry
@limits(calls=5, period=1)
def retry_request(func, retries=5, delay=1):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"Retrying due to {e}, attempt {i+1}")
            time.sleep(delay * (2 ** i))
    return None

# Fetch NASDAQ tickers and download stock data in parallel
def fetch_nasdaq_tickers():
    file_path = os.path.join(CACHE_DIR, 'nasdaqlisted.txt')

    lock = FileLock(file_path + ".lock")

    with lock:
        if not os.path.exists(file_path) or (time.time() - os.path.getmtime(file_path)) > CACHE_TTL:
            ftp = ftplib.FTP(ftp_server)
            ftp.login()

            with open(file_path, 'wb') as file:
                ftp.retrbinary(f'RETR {ftp_path}', file.write)

            ftp.quit()

    try:
        nasdaq_data = pd.read_csv(file_path, sep='|')
        nasdaq_tickers_df = nasdaq_data[nasdaq_data['Test Issue'] == 'N']
        nasdaq_tickers = nasdaq_tickers_df['Symbol'].tolist()
        return nasdaq_tickers
    except pd.errors.EmptyDataError:
        print(f"Error reading NASDAQ data. File {file_path} is empty or invalid.")
        return []

# Get NASDAQ stocks with prices greater than $5 and (Close * Volume) < 1,000,000
def get_nasdaq_stocks_above_5():
    nasdaq_tickers = fetch_nasdaq_tickers()

    stocks_above_5 = []
    stock_data_map = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_stock_data, ticker) for ticker in nasdaq_tickers]
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                stocks_above_5.append(ticker)
                stock_data_map[ticker] = data

    return stocks_above_5, stock_data_map

# Download NASDAQ stock data with a condition for 900 days of data
def download_stock_data(ticker):
    cached_data = load_cached_stock_data(ticker)
    if cached_data is not None:
        return ticker, cached_data

    try:
        data = retry_request(lambda: yf.download(ticker, start=extended_start_date, end=target_date + timedelta(days=1)))

        if not data.empty and len(data) >= 900:
            last_close_price = data['Close'].iloc[-1]
            last_volume = data['Volume'].iloc[-1]
            last_day_product = last_close_price * last_volume

            if last_day_product > 1_000_000 and last_close_price > 5:
                cache_stock_data(ticker, data)
                return ticker, data
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    return ticker, None

# Cache stock data
def cache_stock_data(ticker, data):
    file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    data.to_csv(file_path)

def load_cached_stock_data(ticker):
    file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(file_path) and (time.time() - os.path.getmtime(file_path)) < CACHE_TTL:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    return None

# Fetch S&P 500 stocks
def get_sp500_stocks():
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_tickers = sp500_table[0]['Symbol'].tolist()

    all_stocks = []
    stock_data_map = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_sp500_stock_data, ticker) for ticker in sp500_tickers]
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                all_stocks.append(ticker)
                stock_data_map[ticker] = data

    return all_stocks, stock_data_map

# Download S&P 500 stock data
def download_sp500_stock_data(ticker):
    cached_data = load_cached_stock_data(ticker)
    if cached_data is not None:
        return ticker, cached_data

    try:
        data = retry_request(lambda: yf.download(ticker, start=extended_start_date, end=target_date + timedelta(days=1)))
        if not data.empty:
            cache_stock_data(ticker, data)
            return ticker, data
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    return ticker, None

# Download Saudi stock data
def download_saudi_stock_data(ticker):
    cached_data = load_cached_stock_data(ticker)
    if cached_data is not None:
        return ticker, cached_data

    try:
        data = retry_request(lambda: yf.download(ticker, start=extended_start_date_saudi, end=target_date_saudi + timedelta(days=1)))
        if not data.empty:
            cache_stock_data(ticker, data)
            return ticker, data
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    return ticker, None

# Fetch Saudi stocks
def get_saudi_stocks():
    saudi_tickers = ['1050.SR', '1150.SR', '1210.SR', '2020.SR', '1820.SR', '2040.SR', '2010.SR', '2050.SR', '1080.SR',
                 '1120.SR', '2030.SR', '2110.SR', '2120.SR', '2090.SR', '1810.SR', '2130.SR', '2150.SR', '2170.SR',
                 '2080.SR', '2160.SR', '2180.SR', '2190.SR', '2200.SR', '2210.SR', '2220.SR', '2230.SR', '2060.SR',
                 '2070.SR', '2100.SR', '2280.SR', '1060.SR', '2300.SR', '2240.SR', '1180.SR', '2330.SR', '2340.SR',
                 '2310.SR', '2360.SR', '2380.SR', '2350.SR', '2370.SR', '2320.SR', '3003.SR', '3004.SR', '3005.SR',
                 '4001.SR', '4002.SR', '4004.SR', '4008.SR', '4006.SR', '4030.SR', '4020.SR', '3002.SR', '4061.SR',
                 '4040.SR', '4007.SR', '4050.SR', '4070.SR', '4140.SR', '4110.SR', '4080.SR', '4100.SR', '4130.SR',
                 '4190.SR', '4200.SR', '4210.SR', '4220.SR', '4230.SR', '4240.SR', '4180.SR', '4090.SR', '4250.SR',
                 '4270.SR', '4260.SR', '4300.SR', '4310.SR', '4160.SR', '4340.SR', '4330.SR', '4005.SR', '4290.SR',
                 '4342.SR', '4347.SR', '4344.SR', '4346.SR', '6001.SR', '6010.SR', '6050.SR', '6060.SR', '6070.SR',
                 '4280.SR', '6020.SR', '7020.SR', '6090.SR', '6040.SR', '7010.SR', '7040.SR', '4320.SR', '8030.SR',
                 '8010.SR', '8020.SR', '7200.SR', '8050.SR', '8070.SR', '8060.SR', '8100.SR', '8120.SR', '8180.SR',
                 '8150.SR', '8160.SR', '8190.SR', '8200.SR', '8240.SR', '8230.SR', '8270.SR', '8210.SR', '8250.SR',
                 '8310.SR', '8300.SR', '8170.SR', '9300.SR', '9510.SR', '9540.SR', '9520.SR', '9550.SR', '9570.SR',
                 '9580.SR', '9590.SR', '9560.SR', '2001.SR', '1830.SR', '2140.SR', '2250.SR', '2270.SR', '2290.SR',
                 '4003.SR', '4150.SR', '4170.SR', '4345.SR', '7030.SR', '8040.SR', '8260.SR', '8280.SR', '9400.SR',
                 '9530.SR', '9600.SR']

    all_stocks = []
    stock_data_map = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_saudi_stock_data, ticker) for ticker in saudi_tickers]
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                all_stocks.append(ticker)
                stock_data_map[ticker] = data

    return all_stocks, stock_data_map
# Download CRYPTO stock data
def download_crypto_stock_data(ticker):
    cached_data = load_cached_stock_data(ticker)
    if cached_data is not None:
        return ticker, cached_data

    try:
        data = retry_request(lambda: yf.download(ticker, start=extended_start_date, end=target_date + timedelta(days=1)))
        if not data.empty:
            cache_stock_data(ticker, data)
            return ticker, data
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    return ticker, None

# Fetch crypto stocks
# Function to fetch all tickers using Playwright
async def fetch_all_crypto_tickers():
    url = "https://coinmarketcap.com/exchanges/coinbase-exchange/"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Set headless=False to view the browser
        page = await browser.new_page()
        await page.goto(url)

        tickers = []

        while True:
            # Wait for the 'Load More' button to appear
            load_more_button = await page.query_selector('button:has-text("Load More")')
            if not load_more_button:
                break  # Stop if there is no more 'Load More' button

            # Extract tickers from the current page
            rows = await page.query_selector_all('table tr')
            for row in rows:
                columns = await row.query_selector_all('td')
                if len(columns) > 2:
                    ticker = await columns[2].inner_text()
                    tickers.append(ticker.strip())

            # Click the 'Load More' button to load the next set of tickers
            await load_more_button.click()
            await page.wait_for_timeout(2000)  # Wait for 2 seconds to load more tickers

        await browser.close()
        return tickers

# Function to get the dynamically fetched tickers and download data for them
def get_crypto_stocks():
    # Fetch all tickers using the Playwright logic
    crypto_tickers = asyncio.run(fetch_all_crypto_tickers())
    
    all_stocks = []
    stock_data_map = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_crypto_stock_data, ticker) for ticker in crypto_tickers]
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                all_stocks.append(ticker)
                stock_data_map[ticker] = data

    return all_stocks, stock_data_map


# Calculate RSI using TradingView's RMA formula
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    alpha = 1 / period
    rma_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    rma_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = rma_gain / rma_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Calculate Stochastic RSI
def calculate_stochastic_rsi(data, period=14):
    delta = data['Close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)

    gain_ema = up.ewm(com=period-1, min_periods=period).mean()
    loss_ema = down.ewm(com=period-1, min_periods=period).mean()

    RS = gain_ema / loss_ema
    data['RSI'] = 100 - (100 / (1 + RS))

    data['LL_RSI'] = data['RSI'].rolling(window=period).min()
    data['HH_RSI'] = data['RSI'].rolling(window=period).max()

    data['Stochastic_RSI_K'] = 100 * ((data['RSI'] - data['LL_RSI']) / (data['HH_RSI'] - data['LL_RSI']))
    data['Stochastic_RSI_D'] = data['Stochastic_RSI_K'].rolling(window=3).mean()

    return data['Stochastic_RSI_D']

# Generate PDFs
def generate_pdfs(tickers, market_name, stock_data_map, target_date):
    rsi_less_than_51_pdf = FPDF()
    combined_alert_pdf = FPDF()
    stoch_rsi_less_than_0_5_pdf = FPDF()

    rsi_less_than_51_pdf.add_page()
    combined_alert_pdf.add_page()
    stoch_rsi_less_than_0_5_pdf.add_page()

    rsi_less_than_51_pdf.set_font("Arial", size=10)
    combined_alert_pdf.set_font("Arial", size=10)
    stoch_rsi_less_than_0_5_pdf.set_font("Arial", size=10)

    for ticker in tickers:
        try:
            data = stock_data_map.get(ticker)
            if data is None or data.empty:
                continue

            rsi = calculate_rsi(data, period=14)
            rsi_filtered = rsi.tail(900)
            lowest_rsi_45_months = rsi_filtered.min()
            rsi_value_today = rsi.iloc[-1]

            if rsi_value_today <= 1.51 * lowest_rsi_45_months:
                rsi_less_than_51_pdf.cell(0, 10, f" Stock: {ticker}, RSI_{target_date.strftime('%Y/%m/%d')}: {rsi_value_today:.2f}, Lowest RSI of last 900 TD: {lowest_rsi_45_months:.2f}", ln=True)

            stochastic_rsi = calculate_stochastic_rsi(data, period=14)
            stoch_rsi_d_today = stochastic_rsi.iloc[-1]

            if rsi_value_today <= 1.51 * lowest_rsi_45_months and stoch_rsi_d_today <= 0.5:
                combined_alert_pdf.cell(0, 10, f" Stock: {ticker}, RSI_{target_date.strftime('%Y/%m/%d')} : {rsi_value_today:.2f}, Stochastic RSI_{target_date.strftime('%Y/%m/%d')} : {stoch_rsi_d_today:.2f}", ln=True)

            if stoch_rsi_d_today <= 0.5:
                stoch_rsi_less_than_0_5_pdf.cell(0, 10, f" Stock: {ticker}, Stochastic RSI_{target_date.strftime('%Y/%m/%d')} : {stoch_rsi_d_today:.2f}", ln=True)

        except Exception as e:
            print(f"Error for {market_name} ticker {ticker}: {e}")
            continue

    rsi_less_than_51_file_path = f"{market_name}_RSI_Lessthan_51.pdf"
    combined_alert_file_path = f"{market_name}_combined_Alert.pdf"
    stoch_rsi_less_than_0_5_file_path = f"{market_name}_StochRSI_Lessthan_0_5.pdf"

    rsi_less_than_51_pdf.output(rsi_less_than_51_file_path)
    combined_alert_pdf.output(combined_alert_file_path)
    stoch_rsi_less_than_0_5_pdf.output(stoch_rsi_less_than_0_5_file_path)

    return (rsi_less_than_51_file_path, combined_alert_file_path, stoch_rsi_less_than_0_5_file_path)

# Check if PDFs exist and delete them if older than 14 hours
def check_pdfs_existence_and_delete_old(pdf_files):
    current_time = time.time()
    for file in pdf_files:
        if os.path.exists(file):
            # Get the file's modification time
            file_mod_time = os.path.getmtime(file)
            # Check if the file is older than 14 hours (50000 seconds)
            if current_time - file_mod_time > 50000:
                os.remove(file)  # Delete the file
                print(f"{file} deleted as it is older than 14 hours.")
            else:
                return all(os.path.exists(file) for file in pdf_files)
                print(f"{file} exists and is within 14 hours.")
        else:
            print(f"{file} does not exist.")

# Send PDFs via Telegram
async def send_pdfs(bot, chat_id, pdf_files):
    try:
        for file in pdf_files:
            if os.path.exists(file):
                await bot.send_document(chat_id=chat_id, document=open(file, "rb"))
            else:
                print(f"File {file} does not exist.")
    except Exception as e:
        print(f"An error occurred while sending PDFs: {e}")

# Start command to welcome users
async def start(update: Update, context):
    await update.message.reply_text("Hello! Type 'nasdaq', 'sp500', 'saudi' , 'crypto' or 'all' to receive stock reports.")


# Handle NASDAQ, S&P 500, Saudi, and all stock reports
async def handle_message(update: Update, context):
    message_text = update.message.text.lower()

    # Handling NASDAQ report
    if message_text == "nasdaq":
        nasdaq_pdf_files = ["NASDAQ_RSI_Lessthan_51.pdf", "NASDAQ_combined_Alert.pdf", "NASDAQ_StochRSI_Lessthan_0_5.pdf"]
        if check_pdfs_existence_and_delete_old(nasdaq_pdf_files):
            await update.message.reply_text("Fetching NASDAQ stock reports, please wait...")
            await send_pdfs(context.bot, update.message.chat_id, nasdaq_pdf_files)
        else:
            await update.message.reply_text("Generating NASDAQ stock reports, please wait...")
            nasdaq_tickers, nasdaq_stock_data_map = get_nasdaq_stocks_above_5()
            nasdaq_pdf_files = generate_pdfs(nasdaq_tickers, "NASDAQ", nasdaq_stock_data_map, target_date)
            await send_pdfs(context.bot, update.message.chat_id, nasdaq_pdf_files)

    # Handling S&P 500 report
    elif message_text == "sp500":
        sp500_pdf_files = ["SP500_RSI_Lessthan_51.pdf", "SP500_combined_Alert.pdf", "SP500_StochRSI_Lessthan_0_5.pdf"]
        if check_pdfs_existence_and_delete_old(sp500_pdf_files):
            await update.message.reply_text("Fetching S&P 500 stock reports, please wait...")
            await send_pdfs(context.bot, update.message.chat_id, sp500_pdf_files)
        else:
            await update.message.reply_text("Generating S&P 500 stock reports, please wait...")
            sp500_tickers, sp500_stock_data_map = get_sp500_stocks()
            sp500_pdf_files = generate_pdfs(sp500_tickers, "SP500", sp500_stock_data_map, target_date)
            await send_pdfs(context.bot, update.message.chat_id, sp500_pdf_files)

    # Handling Saudi stock report
    elif message_text == "saudi":
        saudi_pdf_files = ["SAUDI_RSI_Lessthan_51.pdf", "SAUDI_combined_Alert.pdf", "SAUDI_StochRSI_Lessthan_0_5.pdf"]
        if check_pdfs_existence_and_delete_old(saudi_pdf_files):
            await update.message.reply_text("Fetching Saudi stock reports, please wait...")
            await send_pdfs(context.bot, update.message.chat_id, saudi_pdf_files)
        else:
            await update.message.reply_text("Generating Saudi stock reports, please wait...")
            saudi_tickers, saudi_stock_data_map = get_saudi_stocks()
            saudi_pdf_files = generate_pdfs(saudi_tickers, "SAUDI", saudi_stock_data_map, target_date_saudi)
            await send_pdfs(context.bot, update.message.chat_id, saudi_pdf_files)
    # Handling crypto stock report
    elif message_text == "crypto":
        crypto_pdf_files = ["CRYPTO_RSI_Lessthan_51.pdf", "CRYPTO_combined_Alert.pdf", "CRYPTO_StochRSI_Lessthan_1.pdf"]
        if check_pdfs_existence_and_delete_old(crypto_pdf_files):
            await update.message.reply_text("Fetching crypto stock reports, please wait...")
            await send_pdfs(context.bot, update.message.chat_id, crypto_pdf_files)
        else:
            await update.message.reply_text("Generating crypto stock reports, please wait...")
            crypto.generate_pdf("CRYPTO", target_date)  # Call the generate_pdf function from crypto.py
            await send_pdfs(context.bot, update.message.chat_id, crypto_pdf_files)

    # Handling 'all' report (NASDAQ, S&P 500, Saudi and Crypto)
    elif message_text == "all":
        nasdaq_pdf_files = ["NASDAQ_RSI_Lessthan_51.pdf", "NASDAQ_combined_Alert.pdf", "NASDAQ_StochRSI_Lessthan_0_5.pdf"]
        sp500_pdf_files = ["SP500_RSI_Lessthan_51.pdf", "SP500_combined_Alert.pdf", "SP500_StochRSI_Lessthan_0_5.pdf"]
        saudi_pdf_files = ["SAUDI_RSI_Lessthan_51.pdf", "SAUDI_combined_Alert.pdf", "SAUDI_StochRSI_Lessthan_0_5.pdf"]
        crypto_pdf_files = ["CRYPTO_RSI_Lessthan_51.pdf", "CRYPTO_combined_Alert.pdf", "CRYPTO_StochRSI_Lessthan_1.pdf"]

        await update.message.reply_text("Fetching all stock reports, please wait...")

        # Check NASDAQ
        if check_pdfs_existence_and_delete_old(nasdaq_pdf_files):
            await send_pdfs(context.bot, update.message.chat_id, nasdaq_pdf_files)
        else:
            nasdaq_tickers, nasdaq_stock_data_map = get_nasdaq_stocks_above_5()
            nasdaq_pdf_files = generate_pdfs(nasdaq_tickers, "NASDAQ", nasdaq_stock_data_map, target_date)
            await send_pdfs(context.bot, update.message.chat_id, nasdaq_pdf_files)

        # Check S&P 500
        if check_pdfs_existence_and_delete_old(sp500_pdf_files):
            await send_pdfs(context.bot, update.message.chat_id, sp500_pdf_files)
        else:
            sp500_tickers, sp500_stock_data_map = get_sp500_stocks()
            sp500_pdf_files = generate_pdfs(sp500_tickers, "SP500", sp500_stock_data_map, target_date)
            await send_pdfs(context.bot, update.message.chat_id, sp500_pdf_files)

        # Check Saudi
        if check_pdfs_existence_and_delete_old(saudi_pdf_files):
            await send_pdfs(context.bot, update.message.chat_id, saudi_pdf_files)
        else:
            saudi_tickers, saudi_stock_data_map = get_saudi_stocks()
            saudi_pdf_files = generate_pdfs(saudi_tickers, "SAUDI", saudi_stock_data_map, target_date_saudi)
            await send_pdfs(context.bot, update.message.chat_id, saudi_pdf_files)
        # Check CRYPTO
        if check_pdfs_existence_and_delete_old(crypto_pdf_files):
            await send_pdfs(context.bot, update.message.chat_id, crypto_pdf_files)
        else:
            await update.message.reply_text("Generating crypto stock reports, please wait...")
            crypto_pdf_files = crypto.generate_pdf("CRYPTO", target_date)
            await send_pdfs(context.bot, update.message.chat_id, crypto_pdf_files)
# Main function to start the bot and set up handlers
def main():
    try:
        # Create the bot application
        app = Application.builder().token(bot_token).build()

        # Set up handlers
        app.add_handler(CommandHandler("start", start))  # Start command
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))  # Text messages handler

        # Apply nest_asyncio to allow nested event loops in environments like Jupyter
        nest_asyncio.apply()

        # Get the existing event loop or create a new one if none exists
        loop = asyncio.get_event_loop()

        # Run the bot in the existing event loop
        loop.run_until_complete(app.run_polling())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

