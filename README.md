# Hybrid Stock Price and Direction Prediction Model

This project implements two separate deep learning models using LSTMs (Long Short-Term Memory networks) to predict stock market behavior:

1.  **Direction Prediction Model**: A classification model that predicts whether the stock price will go up or down.
2.  **Price Prediction Model**: A regression model that predicts the future closing price of the stock.

This two-model approach allows for specialized architectures and training processes for each task, leading to better performance.

## Project Structure

```
Hybrid-Stock-Price-and-Direction-Prediction-Model/
├── data/
│   ├── ^NSEI_data.csv
│   └── ^NSEI_data_with_features.csv
├── models/
│   ├── best_direction_model.keras
│   └── best_price_model.keras
├── src/
│   ├── data_collection/
│   │   ├── get_data.py
│   │   └── get_news.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── train_direction_model.py
│       └── train_price_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Hybrid-Stock-Price-and-Direction-Prediction-Model
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

First, ensure you have the necessary data by running the data collection and feature engineering scripts.

### 1. Train the Direction Prediction Model

This model predicts the next day's price movement (up or down).

```bash
python src/models/train_direction_model.py
```

### 2. Train the Price Prediction Model

This model predicts the next day's closing price.

```bash
python src/models/train_price_model.py
```

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib / Seaborn
- `imbalanced-learn` for handling class imbalance.