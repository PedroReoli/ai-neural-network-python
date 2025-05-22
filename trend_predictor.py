import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib

class TrendPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = self._build_model()
        
    def _build_model(self):
        """Build an advanced LSTM model for trend prediction"""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, 18))
        
        # LSTM layers with residual connections
        lstm1 = LSTM(128, return_sequences=True)(input_layer)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)
        
        lstm3 = LSTM(32)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(0.3)(lstm3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        # Output layers
        direction_output = Dense(1, activation='sigmoid', name='direction')(dense2)
        intensity_output = Dense(1, activation='sigmoid', name='intensity')(dense2)
        
        # Create model
        model = Model(inputs=input_layer, outputs=[direction_output, intensity_output])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'direction': 'binary_crossentropy',
                'intensity': 'mse'
            },
            metrics={
                'direction': 'accuracy',
                'intensity': 'mae'
            }
        )
        
        return model
    
    def prepare_sequences(self, X):
        """Prepare sequences for LSTM input"""
        sequences = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:(i + self.sequence_length)])
        return np.array(sequences)
    
    def train(self, X, y_direction, y_intensity, epochs=50, batch_size=32):
        """Train the model with advanced callbacks"""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq = self.prepare_sequences(X_scaled)
        y_direction_seq = y_direction[self.sequence_length:]
        y_intensity_seq = y_intensity[self.sequence_length:]
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_seq,
            {
                'direction': y_direction_seq,
                'intensity': y_intensity_seq
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with confidence scores"""
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequences(X_scaled)
        
        # Get predictions
        direction_pred, intensity_pred = self.model.predict(X_seq)
        
        # Calculate confidence scores
        direction_confidence = np.abs(direction_pred - 0.5) * 2
        intensity_confidence = 1 - np.abs(intensity_pred - 0.5) * 2
        
        return {
            'direction': direction_pred,
            'intensity': intensity_pred,
            'direction_confidence': direction_confidence,
            'intensity_confidence': intensity_confidence
        }
    
    def save_model(self, path):
        """Save the model and scaler"""
        self.model.save(f"{path}_model.h5")
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
    
    def load_model(self, path):
        """Load a saved model and scaler"""
        self.model = tf.keras.models.load_model(f"{path}_model.h5")
        self.scaler = joblib.load(f"{path}_scaler.joblib")
    
    def evaluate(self, X, y_direction, y_intensity):
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequences(X_scaled)
        y_direction_seq = y_direction[self.sequence_length:]
        y_intensity_seq = y_intensity[self.sequence_length:]
        
        results = self.model.evaluate(
            X_seq,
            {
                'direction': y_direction_seq,
                'intensity': y_intensity_seq
            }
        )
        
        return {
            'loss': results[0],
            'direction_loss': results[1],
            'intensity_loss': results[2],
            'direction_accuracy': results[3],
            'intensity_mae': results[4]
        } 