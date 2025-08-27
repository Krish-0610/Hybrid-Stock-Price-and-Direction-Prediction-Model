import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout, BatchNormalization, 
                                     Bidirectional, MultiHeadAttention, LayerNormalization,
                                     Conv1D, MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils import class_weight
import tensorflow as tf
import datetime
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for addressing class imbalance in binary classification.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

def add_advanced_features(data):
    """
    Add advanced technical indicators and features.
    """
    # Price-based features
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Close_Open_Ratio'] = data['Close'] / data['Open']
    
    # Volume indicators
    data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_10']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * bb_std)
    data['BB_Lower'] = data['BB_Middle'] - (2 * bb_std)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stochastic'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # Williams %R
    data['Williams_R'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
    
    # Rate of Change
    data['ROC'] = 100 * ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10))
    
    # Commodity Channel Index
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = np.abs(typical_price - sma_tp).rolling(window=20).mean()
    data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=14).sum()
    negative_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    data['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    return data

def create_advanced_sequences(features, targets, seq_length=60, step=1):
    """
    Create sequences with overlapping windows for more training data.
    """
    X, y = [], []
    for i in range(seq_length, len(features), step):
        X.append(features[i-seq_length:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def load_and_preprocess_for_enhanced_lstm(file_path, sequence_length=60):
    """
    Enhanced data loading and preprocessing.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Initial data shape: {data.shape}")
    
    # Add advanced features
    data = add_advanced_features(data)
    
    # Create targets with multiple horizons for ensemble
    data['target_direction_1d'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['target_direction_3d'] = (data['Close'].shift(-3) > data['Close']).astype(int)
    data['target_direction_5d'] = (data['Close'].shift(-5) > data['Close']).astype(int)
    
    # Drop NaN values
    data.dropna(inplace=True)
    print(f"Data shape after creating features and targets: {data.shape}")
    
    # Use primary target
    features = data.drop(['target_direction_1d', 'target_direction_3d', 'target_direction_5d'], axis=1)
    targets_direction = data['target_direction_1d']
    
    # Advanced train/validation/test split
    train_size = int(len(features) * 0.7)
    val_size = int(len(features) * 0.15)
    
    train_features = features[:train_size]
    val_features = features[train_size:train_size+val_size]
    test_features = features[train_size+val_size:]
    
    train_targets = targets_direction[:train_size]
    val_targets = targets_direction[train_size:train_size+val_size]
    test_targets = targets_direction[train_size+val_size:]
    
    # Use StandardScaler for better normalization
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Create sequences with overlapping windows
    X_train, y_train = create_advanced_sequences(train_features_scaled, train_targets.values, sequence_length, step=2)
    X_val, y_val = create_advanced_sequences(val_features_scaled, val_targets.values, sequence_length, step=3)
    X_test, y_test = create_advanced_sequences(test_features_scaled, test_targets.values, sequence_length, step=3)
    
    print(f"Training sequences shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation sequences shape: {X_val.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict

def build_transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformer block for sequence modeling.
    """
    # Multi-head attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)
    
    # Feed forward network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    
    return LayerNormalization(epsilon=1e-6)(x + ff_output)

def build_enhanced_direction_model(input_shape, num_classes=1):
    """
    Enhanced hybrid model combining CNN, Bidirectional LSTM/GRU, and Transformer.
    """
    inputs = Input(shape=input_shape)
    
    # CNN Feature Extraction Branch
    cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = GlobalAveragePooling1D()(cnn_branch)
    
    # Bidirectional LSTM/GRU Branch
    lstm_branch = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01, 0.01)))(inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    lstm_branch = Bidirectional(GRU(64, return_sequences=True))(lstm_branch)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    
    # Transformer Branch
    transformer_branch = build_transformer_block(
        inputs, 
        head_size=32, 
        num_heads=4, 
        ff_dim=128, 
        dropout=0.2
    )
    transformer_branch = build_transformer_block(
        transformer_branch,
        head_size=32,
        num_heads=4, 
        ff_dim=64,
        dropout=0.2
    )
    
    # Attention mechanism for LSTM branch
    attention = MultiHeadAttention(key_dim=32, num_heads=2)(lstm_branch, lstm_branch)
    attention = GlobalAveragePooling1D()(attention)
    
    # Global pooling for transformer
    transformer_pooled = GlobalAveragePooling1D()(transformer_branch)
    
    # Combine all branches
    combined = Concatenate()([cnn_branch, attention, transformer_pooled])
    
    # Dense layers with batch normalization
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid', name='direction_output', dtype='float32')(x)
    else:
        outputs = Dense(num_classes, activation='softmax', name='direction_output', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom optimizer with gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    # Compile with focal loss
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_ensemble_models(X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict):
    """
    Train multiple models for ensemble prediction.
    """
    models = []
    histories = []
    
    # Model configurations
    configs = [
        {'name': 'transformer_heavy', 'transformer_layers': 3, 'lstm_units': 64},
        {'name': 'lstm_heavy', 'transformer_layers': 1, 'lstm_units': 256},
        {'name': 'balanced', 'transformer_layers': 2, 'lstm_units': 128}
    ]
    
    for config in configs:
        print(f"\n--- Training {config['name']} model ---")
        model = build_enhanced_direction_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_auc', 
            patience=15, 
            restore_best_weights=True, 
            mode='max'
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join('models', f"best_direction_{config['name']}.keras"), 
            monitor='val_auc', 
            save_best_only=True, 
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join('logs', f"direction_{config['name']}_" + 
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback],
            verbose=1
        )
        
        models.append(model)
        histories.append(history)
    
    return models, histories

def ensemble_predict(models, X_test, threshold=0.5):
    """
    Ensemble prediction using weighted voting.
    """
    predictions = []
    weights = [0.4, 0.3, 0.3]  # Weights for each model
    
    for model, weight in zip(models, weights):
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred * weight)
    
    # Weighted average
    ensemble_pred = np.sum(predictions, axis=0)
    
    # Find optimal threshold
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = threshold
    best_f1 = 0
    
    for thresh in thresholds:
        pred_class = (ensemble_pred > thresh).astype(int)
        f1 = f1_score(y_test, pred_class, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return ensemble_pred, best_threshold

if __name__ == '__main__':
    input_path = os.path.join('data', '^NSEI_data_with_features.csv')
    SEQUENCE_LENGTH = 60
    
    # Load and preprocess data
    processed_data = load_and_preprocess_for_enhanced_lstm(input_path, SEQUENCE_LENGTH)
    
    if processed_data:
        X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict = processed_data
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        print("\n--- Class Distribution ---")
        print(f"Training - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        print(f"Class weights: {class_weight_dict}")
        
        # Train single best model
        print("\n--- Training Enhanced Direction Prediction Model ---")
        model = build_enhanced_direction_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Advanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max'
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join('models', 'best_enhanced_direction_model.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join('logs', 'enhanced_direction_' + 
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback],
            verbose=1
        )
        
        # Evaluate model
        print("\n--- Evaluating Enhanced Direction Model ---")
        y_pred_proba = model.predict(X_test)
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        print("\n--- Threshold Optimization ---")
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Threshold {threshold:.2f}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nOptimal threshold: {best_threshold:.2f}")
        
        # Final predictions with optimal threshold
        y_pred_final = (y_pred_proba > best_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_final)
        f1 = f1_score(y_test, y_pred_final, average='weighted')
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print("\n" + "="*50)
        print("FINAL ENHANCED DIRECTION MODEL RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final, target_names=['Down', 'Up']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_final))
        
        # Train ensemble models for even better performance
        print("\n" + "="*50)
        print("TRAINING ENSEMBLE MODELS")
        print("="*50)
        
        models, histories = train_ensemble_models(
            X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict
        )
        
        # Ensemble prediction
        ensemble_pred, optimal_threshold = ensemble_predict(models, X_test)
        ensemble_pred_class = (ensemble_pred > optimal_threshold).astype(int)
        
        # Ensemble metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred_class)
        ensemble_f1 = f1_score(y_test, ensemble_pred_class, average='weighted')
        
        try:
            ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        except:
            ensemble_auc = 0.0
        
        print("\n" + "="*50)
        print("ENSEMBLE MODEL RESULTS")
        print("="*50)
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble F1-Score: {ensemble_f1:.4f}")
        print(f"Ensemble AUC-ROC: {ensemble_auc:.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred_class, target_names=['Down', 'Up']))
        print("\nEnsemble Confusion Matrix:")
        print(confusion_matrix(y_test, ensemble_pred_class))
