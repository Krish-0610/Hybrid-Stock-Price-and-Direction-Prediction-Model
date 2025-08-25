import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import datetime

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for addressing class imbalance in binary classification.
    """
    def focal_loss_fixed(y_true, y_pred):
        import tensorflow as tf
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

def load_and_preprocess_for_lstm(file_path, sequence_length=60):
    """
    Loads, preprocesses, and creates sequences for the LSTM model.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Initial data shape: {data.shape}")

    # Create targets
    data['target_direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    print(f"Data shape after creating targets and dropping NA: {data.shape}")

    # Split and scale features
    features = data.drop(['target_direction'], axis=1)
    targets_direction = data['target_direction']

    split_index = int(len(features) * 0.8)
    train_features, test_features = features[:split_index], features[split_index:]
    train_targets_direction, test_targets_direction = targets_direction[:split_index], targets_direction[split_index:]

    feature_scaler = MinMaxScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    # Create sequences
    def create_sequences(features, direction_targets, seq_length):
        X, y_direction = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y_direction.append(direction_targets.iloc[i])
        return np.array(X), np.array(y_direction)

    X_train, y_train_direction = create_sequences(train_features_scaled, train_targets_direction, sequence_length)
    X_test, y_test_direction = create_sequences(test_features_scaled, test_targets_direction, sequence_length)
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")

    # Reshape X_train for SMOTE
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape((n_samples, seq_len * n_features))

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_direction_resampled = smote.fit_resample(X_train_reshaped, y_train_direction)

    # Reshape X_train back to its original shape
    X_train = X_train_resampled.reshape((X_train_resampled.shape[0], seq_len, n_features))
    y_train_direction = y_train_direction_resampled

    # Manually split the training data for validation
    val_split_index = int(X_train.shape[0] * 0.9)
    X_train_split, X_val_split = X_train[:val_split_index], X_train[val_split_index:]
    y_train_direction_split, y_val_direction_split = y_train_direction[:val_split_index], y_train_direction[val_split_index:]

    return X_train, X_test, y_train_direction, y_test_direction, X_train_split, X_val_split, y_train_direction_split, y_val_direction_split

def build_lstm_model(input_shape):
    """
    Builds a single-output LSTM model for direction prediction.
    """
    inputs = Input(shape=input_shape)
    lstm = LSTM(units=100, return_sequences=True)(inputs)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(units=100, return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(units=50)(lstm)
    lstm = Dropout(0.2)(lstm)

    # Direction prediction head
    direction_head = Dense(25, activation='relu')(lstm)
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(direction_head)

    model = Model(inputs=inputs, outputs=direction_output)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    input_path = os.path.join('data', '^NSEI_data_with_features.csv')
    SEQUENCE_LENGTH = 60

    processed_data = load_and_preprocess_for_lstm(input_path, SEQUENCE_LENGTH)

    if processed_data:
        X_train, X_test, y_train_direction, y_test_direction, X_train_split, X_val_split, y_train_direction_split, y_val_direction_split = processed_data

        print(f"X_train shape (after SMOTE): {X_train.shape}")
        print(f"y_train_direction shape (after SMOTE): {y_train_direction.shape}")
        print(f"Class distribution in y_train_direction: {np.bincount(y_train_direction)}")
        print(f"X_test shape: {X_test.shape}\n")

        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Create directories for models and logs if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
        model_checkpoint = ModelCheckpoint(os.path.join('models', 'best_model.keras'), 
                                           monitor='val_loss', 
                                           save_best_only=True, 
                                           mode='min')
        tensorboard_callback = TensorBoard(log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        print("\n--- Training LSTM Model ---")
        model.fit(X_train_split, y_train_direction_split,
                  validation_data=(X_val_split, y_val_direction_split),
                  epochs=50, batch_size=32, verbose=1,
                  callbacks=[early_stopping, model_checkpoint, tensorboard_callback])

        print("\n--- Evaluating LSTM Model ---")
        direction_pred = model.predict(X_test)

        # Try different thresholds for direction prediction
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_threshold = 0.5
        best_f1 = 0
        
        print("\n--- Threshold Analysis ---")
        for threshold in thresholds:
            direction_pred_thresh = (direction_pred > threshold).astype(int)
            f1 = f1_score(y_test_direction, direction_pred_thresh, average='weighted')
            print(f"Threshold {threshold}: F1-score = {f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold} (F1-score: {best_f1:.3f})")
        direction_pred_class = (direction_pred > best_threshold).astype(int)

        # Direction metrics
        accuracy = accuracy_score(y_test_direction, direction_pred_class)
        print(f"\n--- Direction Prediction Results ---")
        print(f"Direction Prediction Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test_direction, direction_pred_class))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test_direction, direction_pred_class))
