import librosa
import numpy as np
import os
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def extract_mfcc(file_path, n_mfcc=13, max_len=180):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Create sample audio files for demonstration
def create_sample_audio():
    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    emotions = ['happy', 'angry', 'sad']
    for i, emotion in enumerate(emotions):
        freq = 200 + (i * 100)
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        signal += 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
        sf.write(f'sample_{emotion}.wav', signal, sample_rate)
        print(f"Created sample: sample_{emotion}.wav")

# Comment out if you have real audio files
# create_sample_audio()

# For demonstration with sample files
audio_files = []
emotion_labels = []

# If you have real dataset, use this pattern
# for file in os.listdir('path_to_audio_files'):
#     if file.endswith('.wav'):
#         audio_files.append(file)
#         emotion_labels.append(extract_emotion_from_filename(file))

# Using sample files for demonstration
for emotion in ['happy', 'angry', 'sad']:
    for _ in range(10):  # Generate multiple samples
        audio_files.append(f'sample_{emotion}.wav')
        emotion_labels.append(emotion)

# Extract features
X = []
y = []
for file, label in zip(audio_files, emotion_labels):
    mfcc = extract_mfcc(file)
    if mfcc is not None:
        X.append(mfcc)
        y.append(label)

# Prepare data
X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# CNN Model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = cnn_model.fit(X_train, y_train, 
                        epochs=30, 
                        batch_size=16, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=1)

test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {test_acc:.4f}")

# Alternative LSTM Model
lstm_model = Sequential([
    Reshape((X.shape[1], X.shape[2]), input_shape=(X.shape[1], X.shape[2], 1)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_lstm = lstm_model.fit(X_train, y_train,
                              epochs=20,
                              batch_size=16,
                              validation_data=(X_test, y_test),
                              callbacks=[early_stop],
                              verbose=1)

test_loss_lstm, test_acc_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {test_acc_lstm:.4f}")

# Save models
cnn_model.save('emotion_cnn_model.h5')
lstm_model.save('emotion_lstm_model.h5')