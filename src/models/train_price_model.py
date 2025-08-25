import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

def load_and_preprocess_for_price_lstm(file_path, sequence_length=60):
    """
    Loads, preprocesses, and creates sequences for the LSTM price prediction model.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Create target
    data['target_price'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Split and scale features
    features = data.drop(['target_price'], axis=1)
    targets_price = data['target_price']

    split_index = int(len(features) * 0.8)
    train_features, test_features = features[:split_index], features[split_index:]
    train_targets_price, test_targets_price = targets_price[:split_index], targets_price[split_index:]

    feature_scaler = MinMaxScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    # Scale the price target
    price_scaler = MinMaxScaler()
    train_targets_price_scaled = price_scaler.fit_transform(train_targets_price.values.reshape(-1, 1))
    test_targets_price_scaled = price_scaler.transform(test_targets_price.values.reshape(-1, 1))

    # Create sequences
    def create_sequences(features, price_targets, seq_length):
        X, y_price = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y_price.append(price_targets[i])
        return np.array(X), np.array(y_price)

    X_train, y_train_price = create_sequences(train_features_scaled, train_targets_price_scaled, sequence_length)
    X_test, y_test_price = create_sequences(test_features_scaled, test_targets_price_scaled, sequence_length)

    # Manually split the training data for validation
    val_split_index = int(X_train.shape[0] * 0.9)
    X_train_split, X_val_split = X_train[:val_split_index], X_train[val_split_index:]
    y_train_price_split, y_val_price_split = y_train_price[:val_split_index], y_train_price[val_split_index:]

    return X_train, X_test, y_train_price, y_test_price, price_scaler, X_train_split, X_val_split, y_train_price_split, y_val_price_split

def build_price_lstm_model(input_shape):
    """
    Builds a single-output LSTM model for price prediction.
    """
    inputs = Input(shape=input_shape)
    lstm = LSTM(units=100, return_sequences=True)(inputs)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(units=100, return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(units=50)(lstm)
    lstm = Dropout(0.2)(lstm)

    # Price prediction head
    price_head = Dense(25, activation='relu')(lstm)
    price_output = Dense(1, name='price_output')(price_head) # Linear activation is default

    model = Model(inputs=inputs, outputs=price_output)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

if __name__ == '__main__':
    input_path = os.path.join('data', '^NSEI_data_with_features.csv')
    SEQUENCE_LENGTH = 60

    processed_data = load_and_preprocess_for_price_lstm(input_path, SEQUENCE_LENGTH)

    if processed_data:
        X_train, X_test, y_train_price, y_test_price, price_scaler, X_train_split, X_val_split, y_train_price_split, y_val_price_split = processed_data

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}\n")

        model = build_price_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.summary()

        # Create directories for saving models and logs
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
        model_checkpoint = ModelCheckpoint(os.path.join('models', 'best_price_model.keras'), 
                                           monitor='val_loss', 
                                           save_best_only=True, 
                                           mode='min')
        tensorboard_callback = TensorBoard(log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        print("\n--- Training Price Prediction LSTM Model ---")
        model.fit(X_train_split, y_train_price_split,
                  validation_data=(X_val_split, y_val_price_split),
                  epochs=50, batch_size=32, verbose=1,
                  callbacks=[early_stopping, model_checkpoint, tensorboard_callback])

        print("\n--- Evaluating Price Prediction Model ---")
        price_pred_scaled = model.predict(X_test)

        # Inverse transform the price predictions to get the actual values
        price_pred = price_scaler.inverse_transform(price_pred_scaled)
        y_test_price_actual = price_scaler.inverse_transform(y_test_price)

        # Price metrics
        mae = mean_absolute_error(y_test_price_actual, price_pred)
        r2 = r2_score(y_test_price_actual, price_pred)
        print(f"\n--- Price Prediction Results ---")
        print(f"Price Prediction MAE: {mae:.2f}")
        print(f"Price Prediction R-squared: {r2:.4f}")
        
        # Calculate additional price metrics
        mape = np.mean(np.abs((y_test_price_actual - price_pred) / y_test_price_actual)) * 100
        print(f"Price Prediction MAPE: {mape:.2f}%")
