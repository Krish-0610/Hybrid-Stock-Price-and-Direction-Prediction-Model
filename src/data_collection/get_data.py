import yfinance as yf
import pandas as pd
import os

def get_historical_data(ticker, start_date, end_date, output_dir='.\.\data'):
    """
    Downloads historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the data (YYYY-MM-DD).
        end_date (str): The end date for the data (YYYY-MM-DD).
        output_dir (str): The directory to save the data in.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
        return

    # Save the data to a CSV file
    file_path = os.path.join(output_dir, f'{ticker}_data.csv')
    data.to_csv(file_path)
    print(f'Successfully downloaded data for {ticker} and saved to {file_path}')

if __name__ == '__main__':
    # NIFTY 50 ticker in Yahoo Finance is ^NSEI
    NIFTY50_TICKER = '^NSEI'
    START_DATE = '2010-01-01'
    END_DATE = '2025-01-01'

    get_historical_data(NIFTY50_TICKER, START_DATE, END_DATE)
