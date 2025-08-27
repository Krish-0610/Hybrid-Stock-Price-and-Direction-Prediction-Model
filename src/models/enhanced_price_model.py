import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout, BatchNormalization,
                                     Bidirectional, MultiHeadAttention, LayerNormalization,
                                     Conv1D, MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling1D,
                                     Add, Multiply, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import datetime
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def add_advanced_price_features(data):
    """
    Add advanced technical indicators specifically for price prediction.
    """
    # Price momentum features
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Momentum_3'] = data['Close'].pct_change(3)
    data['Price_Momentum_5'] = data['Close'].pct_change(5)
    data['Price_Momentum_10'] = data['Close'].pct_change(10)
    
    # Price volatility
    data['Price_Volatility_5'] = data['Close'].rolling(window=5).std()
    data['Price_Volatility_10'] = data['Close'].rolling(window=10).std()
    data['Price_Volatility_20'] = data['Close'].rolling(window=20).std()
    
    # VWAP (Volume Weighted Average Price)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).rolling(window=14).sum() / data['Volume'].rolling(window=14).sum()
    data['Price_VWAP_Ratio'] = data['Close'] / data['VWAP']
    
    # Fibonacci retracement levels
    period = 20
    high_period = data['High'].rolling(window=period).max()
    low_period = data['Low'].rolling(window=period).min()
    diff = high_period - low_period
    
    data['Fib_23.6'] = high_period - 0.236 * diff
    data['Fib_38.2'] = high_period - 0.382 * diff
    data['Fib_50.0'] = high_period - 0.5 * diff
    data['Fib_61.8'] = high_period - 0.618 * diff
    
    # Distance from Fibonacci levels
    data['Dist_Fib_23.6'] = (data['Close'] - data['Fib_23.6']) / data['Close']
    data['Dist_Fib_38.2'] = (data['Close'] - data['Fib_38.2']) / data['Close']
    data['Dist_Fib_50.0'] = (data['Close'] - data['Fib_50.0']) / data['Close']
    data['Dist_Fib_61.8'] = (data['Close'] - data['Fib_61.8']) / data['Close']
    
    # Pivot points
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['R1'] = 2 * data['Pivot'] - data['Low']
    data['S1'] = 2 * data['Pivot'] - data['High']
    data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
    
    # Distance from pivot levels
    data['Dist_Pivot'] = (data['Close'] - data['Pivot']) / data['Close']
    data['Dist_R1'] = (data['Close'] - data['R1']) / data['Close']
    data['Dist_S1'] = (data['Close'] - data['S1']) / data['Close']
    
    # Ichimoku Cloud Components
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    data['Tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    data['Kijun_sen'] = (high_26 + low_26) / 2
    
    # Elder Ray Index
    ema_13 = data['Close'].ewm(span=13, adjust=False).mean()
    data['Bull_Power'] = data['High'] - ema_13
    data['Bear_Power'] = data['Low'] - ema_13
    
    # On-Balance Volume (OBV)
    obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                   np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
    data['OBV'] = np.cumsum(obv)
    data['OBV_MA'] = data['OBV'].rolling(window=20).mean()
    
    # Price Rate of Change (Multiple Periods)
    for period in [5, 10, 20, 50]:
        data[f'ROC_{period}'] = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100
    
    # Weighted Moving Averages
    weights = np.arange(1, 11)
    data['WMA_10'] = data['Close'].rolling(window=10).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
    
    return data

def load_and_preprocess_for_enhanced_price_lstm(file_path, sequence_length=60):
    """
    Enhanced data loading and preprocessing for price prediction.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Initial data shape: {data.shape}")
    
    # Add all advanced features
    data = add_advanced_price_features(data)
    
    # Create price target
    data['target_price'] = data['Close'].shift(-1)
    
    # Drop NaN values
    data.dropna(inplace=True)
    print(f"Data shape after creating features and targets: {data.shape}")
    
    # Primary target
    features = data.drop(['target_price'], axis=1)
    target_price = data['target_price'].values
    
    # Advanced train/validation/test split
    train_size = int(len(features) * 0.7)
    val_size = int(len(features) * 0.15)
    
    train_features = features[:train_size]
    val_features = features[train_size:train_size+val_size]
    test_features = features[train_size+val_size:]
    
    train_target = target_price[:train_size]
    val_target = target_price[train_size:train_size+val_size]
    test_target = target_price[train_size+val_size:]
    
    # Use RobustScaler for features (handles outliers better)
    feature_scaler = RobustScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    val_features_scaled = feature_scaler.transform(val_features)
    test_features_scaled = feature_scaler.transform(test_features)
    
    # Use MinMaxScaler for targets
    price_scaler = MinMaxScaler()
    train_target_scaled = price_scaler.fit_transform(train_target.reshape(-1, 1))
    val_target_scaled = price_scaler.transform(val_target.reshape(-1, 1))
    test_target_scaled = price_scaler.transform(test_target.reshape(-1, 1))
    
    # Create sequences
    def create_sequences(features, targets, seq_length):
        X, y = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y.append(targets[i])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, sequence_length)
    X_val, y_val = create_sequences(val_features_scaled, val_target_scaled, sequence_length)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, sequence_length)
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Validation sequences shape: {X_val.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, price_scaler

def attention_mechanism(inputs):
    """
    Custom attention mechanism for sequence modeling.
    """
    attention = MultiHeadAttention(key_dim=64, num_heads=4)(inputs, inputs)
    attention = LayerNormalization()(attention)
    return Add()([inputs, attention])

def build_enhanced_price_model(input_shape):
    """
    Enhanced price prediction model with advanced architecture.
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional Feature Extractor
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='causal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(conv2)
    conv3 = BatchNormalization()(conv3)
    
    # Skip connection
    conv_skip = Conv1D(filters=128, kernel_size=1)(inputs)
    conv_out = Add()([conv3, conv_skip])
    
    # Bidirectional LSTM with attention
    lstm1 = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(0.001, 0.001)))(conv_out)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    lstm1 = attention_mechanism(lstm1)
    
    # Stacked GRU layers
    gru1 = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l1_l2(0.001, 0.001)))(lstm1)
    gru1 = BatchNormalization()(gru1)
    gru1 = Dropout(0.3)(gru1)
    
    gru2 = Bidirectional(GRU(64, return_sequences=True))(gru1)
    gru2 = BatchNormalization()(gru2)
    gru2 = Dropout(0.2)(gru2)
    
    # Transformer blocks
    transformer1 = MultiHeadAttention(key_dim=64, num_heads=8)(gru2, gru2)
    transformer1 = LayerNormalization()(transformer1)
    transformer1 = Add()([gru2, transformer1])
    
    ff1 = Dense(256, activation='relu')(transformer1)
    ff1 = Dense(transformer1.shape[-1])(ff1)
    ff1 = Dropout(0.2)(ff1)
    transformer1 = LayerNormalization()(Add()([transformer1, ff1]))
    
    # Second transformer block
    transformer2 = MultiHeadAttention(key_dim=32, num_heads=4)(transformer1, transformer1)
    transformer2 = LayerNormalization()(transformer2)
    transformer2 = Add()([transformer1, transformer2])
    
    ff2 = Dense(128, activation='relu')(transformer2)
    ff2 = Dense(transformer2.shape[-1])(ff2)
    ff2 = Dropout(0.2)(ff2)
    transformer2 = LayerNormalization()(Add()([transformer2, ff2]))
    
    # Global pooling strategies
    avg_pool = GlobalAveragePooling1D()(transformer2)
    max_pool = tf.reduce_max(transformer2, axis=1)
    last_step = Lambda(lambda x: x[:, -1, :])(transformer2)
    
    # Combine different pooling strategies
    combined = Concatenate()([avg_pool, max_pool, last_step])
    
    # Dense layers
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output layer
    price_output = Dense(1, name='price_output', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=price_output)
    
    # Custom optimizer with gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    # Compile with huber loss (robust to outliers)
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

if __name__ == '__main__':
    input_path = os.path.join('data', '^NSEI_data_with_features.csv')
    SEQUENCE_LENGTH = 60
    
    # Load and preprocess data
    processed_data = load_and_preprocess_for_enhanced_price_lstm(input_path, SEQUENCE_LENGTH)
    
    if processed_data:
        X_train, X_val, X_test, y_train, y_val, y_test, price_scaler = processed_data
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        print("\n--- Training Enhanced Price Prediction Model ---")
        
        # Build and train main model
        model = build_enhanced_price_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Print model architecture
        print("\nModel Architecture:")
        model.summary()
        
        # Advanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            mode='min'
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join('models', 'best_enhanced_price_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join('logs', 'enhanced_price_' + 
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback],
            verbose=1
        )
        
        # Evaluate model
        print("\n--- Evaluating Enhanced Price Model ---")
        price_pred_scaled = model.predict(X_test)
        
        # Inverse transform predictions
        price_pred = price_scaler.inverse_transform(price_pred_scaled)
        y_test_actual = price_scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, price_pred)
        mse = mean_squared_error(y_test_actual, price_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, price_pred)
        mape = np.mean(np.abs((y_test_actual - price_pred) / y_test_actual)) * 100
        
        # Direction accuracy
        actual_direction = np.diff(y_test_actual.flatten()) > 0
        pred_direction = np.diff(price_pred.flatten()) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        print("\n" + "="*50)
        print("ENHANCED PRICE MODEL RESULTS")
        print("="*50)
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.4f}")
        
        # Train multiple models for ensemble
        print("\n" + "="*50)
        print("TRAINING ENSEMBLE MODELS")
        print("="*50)
        
        ensemble_predictions = []
        for i in range(3):
            print(f"\n--- Training ensemble model {i+1}/3 ---")
            
            # Set different random seed
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)
            
            # Build model
            ensemble_model = build_enhanced_price_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Train with shorter epochs for ensemble
            ensemble_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=75,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Get predictions
            pred = ensemble_model.predict(X_test, verbose=0)
            ensemble_predictions.append(pred)
        
        # Weighted average ensemble
        weights = [0.4, 0.3, 0.3]
        ensemble_pred_scaled = np.average(ensemble_predictions, axis=0, weights=weights)
        ensemble_pred = price_scaler.inverse_transform(ensemble_pred_scaled)
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_test_actual, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test_actual, ensemble_pred)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_r2 = r2_score(y_test_actual, ensemble_pred)
        ensemble_mape = np.mean(np.abs((y_test_actual - ensemble_pred) / y_test_actual)) * 100
        
        # Ensemble direction accuracy
        ensemble_direction = np.diff(ensemble_pred.flatten()) > 0
        ensemble_dir_accuracy = np.mean(actual_direction == ensemble_direction)
        
        print("\n" + "="*50)
        print("ENSEMBLE MODEL RESULTS")
        print("="*50)
        print(f"Ensemble MAE: {ensemble_mae:.2f}")
        print(f"Ensemble MSE: {ensemble_mse:.2f}")
        print(f"Ensemble RMSE: {ensemble_rmse:.2f}")
        print(f"Ensemble R-squared: {ensemble_r2:.4f}")
        print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
        print(f"Ensemble Direction Accuracy: {ensemble_dir_accuracy:.4f}")
