import pandas as pd
import os

def calculate_technical_indicators(data):
    """
    Calculates technical indicators for the given stock data.

    Args:
        data (pd.DataFrame): DataFrame with historical stock data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    # Simple Moving Averages (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaN values created by the indicators
    data.dropna(inplace=True)

    return data

if __name__ == '__main__':
    # Define file paths relative to the project root
    input_path = os.path.join('data', '^NSEI_data.csv')
    output_path = os.path.join('data', '^NSEI_data_with_features.csv')

    # Load the data
    if os.path.exists(input_path):
        # The CSV has 2 metadata rows and a blank row to skip.
        # The first column is the date index, and the first row is the header.
        stock_data = pd.read_csv(input_path, skiprows=3, index_col=0, parse_dates=True)
        # The header from the file is on the first line, let's read it and apply it.
        header = pd.read_csv(input_path, nrows=0).columns.tolist()
        stock_data.columns = header[1:] # The first column name 'Price' is for the date index
        
        # Calculate indicators
        featured_data = calculate_technical_indicators(stock_data)
        
        # Save the new data
        featured_data.to_csv(output_path)
        print(f'Successfully added features and saved data to {output_path}')
    else:
        print(f'Error: Data file not found at {input_path}')
