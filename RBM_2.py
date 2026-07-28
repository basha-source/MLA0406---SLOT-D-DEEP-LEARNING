from sklearn.neural_network import BernoulliRBM
import numpy as np

# Users × Movies
# M1=Avengers, M2=Spider-Man, M3=Frozen, M4=Toy Story

ratings = np.array([
    [1, 1, 0, 0],  # User 1
    [1, 1, 0, 1],  # User 2
    [0, 0, 1, 1],  # User 3
    [1, 0, 0, 0],  # User 4
])

rbm = BernoulliRBM(n_components=2, random_state=42)
rbm.fit(ratings)

hidden_features = rbm.transform(ratings)

print("Hidden Features Learned:")
print(hidden_features)