import numpy as np

# Student marks in 4 subjects
marks = np.array([[85, 78, 90, 88]])

print("Original Data:")
print(marks)

# -------- Encoder --------
# Compress 4 features into 2 features
encoder_weights = np.array([
    [0.2, 0.4],
    [0.3, 0.1],
    [0.4, 0.3],
    [0.1, 0.2]
])

encoded = np.dot(marks, encoder_weights)

print("\nEncoded Data (2 Features):")
print(encoded)

# -------- Decoder --------
# Reconstruct 2 features back to 4 features
decoder_weights = np.array([
    [0.5, 0.2, 0.3, 0.4],
    [0.3, 0.4, 0.2, 0.1]
])

reconstructed = np.dot(encoded, decoder_weights)

print("\nReconstructed Data:")
print(np.round(reconstructed))