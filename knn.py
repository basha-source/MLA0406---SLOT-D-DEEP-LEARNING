from sklearn.neighbors import KNeighborsClassifier

# Sample data: [Study_Hours, Attendance] -> Result (0=Fail, 1=Pass)
X = [[1, 40], [3, 50], [5, 65], [7, 80], [9, 90]]
y = [0, 0, 1, 1, 1]

# Create and train the KNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predict for a new student
new_student = [[6, 75]]
prediction = knn.predict(new_student)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")