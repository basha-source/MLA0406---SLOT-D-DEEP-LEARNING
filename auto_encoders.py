import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Student marks dataset
X = np.array([
    [85, 90, 88, 80, 92],
    [70, 75, 72, 78, 80],
    [95, 92, 96, 90, 94],
    [60, 65, 62, 70, 68],
    [88, 85, 90, 82, 91],
    [72, 78, 75, 76, 79] 
])

# Normalize data (0-1 range)
X = X / 100.0

# -------------------
# Encoder
# -------------------
input_layer = Input(shape=(5,))

encoded = Dense(3, activation='relu')(input_layer)

# Bottleneck Layer
bottleneck = Dense(2, activation='relu')(encoded)

# -------------------
# Decoder
# -------------------
decoded = Dense(3, activation='relu')(bottleneck)
output_layer = Dense(5, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_layer, output_layer)

# Compile
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# Train
autoencoder.fit(
    X,
    X,
    epochs=200,
    verbose=1
)

# Reconstruct Marks
reconstructed = autoencoder.predict(X)

print("\nOriginal Marks:")
print(X[0] * 100)

print("\nReconstructed Marks:")
print(reconstructed[0] * 100)