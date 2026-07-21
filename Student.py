from sklearn.neural_network import BernoulliRBM
import numpy as np

ratings = np.array([
    [1, 1, 0, 0],  
    [1, 1, 0, 1],
    [0, 0, 1, 1],
])

rbm = BernoulliRBM(n_components=2, random_state=42)
rbm.fit(ratings)

hidden_features = rbm.transform(ratings)

print("Hidden Features Learned:")
print(hidden_features)