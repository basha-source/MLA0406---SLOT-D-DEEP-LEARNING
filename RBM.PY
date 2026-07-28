from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

rbm = BernoulliRBM(n_components=50, random_state=0)
rbm.fit(X)

print("RBM Training Completed")