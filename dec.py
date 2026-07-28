import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# ---------------------------------------------------------
# 1. Create the dataset (50 students)
#    NOTE: This is SYNTHETIC data generated to follow the
#    same pattern as your original 5-row sample (more study
#    hours / attendance / assignment marks -> more likely to
#    pass). Replace this block with pd.read_csv("your_file.csv")
#    if/when you have a real 50-student dataset.
# ---------------------------------------------------------
np.random.seed(42)
n_students = 50

study_hours = np.round(np.random.uniform(1, 10, n_students), 1)
attendance = np.round(np.random.uniform(30, 100, n_students), 0)
assignment = np.round(np.random.uniform(30, 100, n_students), 0)

score = (study_hours / 10) * 0.4 + (attendance / 100) * 0.3 + (assignment / 100) * 0.3
score += np.random.normal(0, 0.05, n_students)
result = (score > 0.55).astype(int)

data = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment": assignment,
    "Result": result
})

data.to_csv("student_performance_50.csv", index=False)
print("Loaded dataset with 50 students.")
print(f"Class balance -> Pass: {result.sum()}, Fail: {len(result) - result.sum()}\n")

# ---------------------------------------------------------
# 2. Split into X and y
# ---------------------------------------------------------
FEATURES = ["Study_Hours", "Attendance", "Assignment"]
X = data[FEATURES]
y = data["Result"]

# ---------------------------------------------------------
# 3. Train the DecisionTreeClassifier
# ---------------------------------------------------------
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
model.fit(X, y)
print(f"Model trained. Training accuracy: {model.score(X, y):.2%}\n")

# ---------------------------------------------------------
# 4. Take a NEW student's details from the user and predict
# ---------------------------------------------------------
print("Enter the new student's details to predict Pass/Fail:")
study_hours_in = float(input("Study Hours: "))
attendance_in = float(input("Attendance: "))
assignment_in = float(input("Assignment Marks: "))

sample = pd.DataFrame(
    [[study_hours_in, attendance_in, assignment_in]], columns=FEATURES
)

prediction = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

print("\n================ PREDICTION ================")
print(f"Student -> Study_Hours={study_hours_in}, Attendance={attendance_in}, Assignment={assignment_in}")
print(f"Result: {'PASS' if prediction == 1 else 'FAIL'}")
print(f"Confidence: Fail={proba[0]:.1%}, Pass={proba[1]:.1%}")
print("==============================================")

# ---------------------------------------------------------
# 5. Explain the prediction: show the exact path the tree
#    took through its questions to reach that decision
# ---------------------------------------------------------
print("\n--- Why the model predicted this ---")
feature_names = np.array(FEATURES)
node_indicator = model.decision_path(sample)
leaf_id = model.apply(sample)
node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]

for node_id in node_index:
    if leaf_id[0] == node_id:
        print(f"-> Reached leaf node {node_id}: final prediction = "
              f"{'PASS' if prediction == 1 else 'FAIL'}")
        continue

    feature = feature_names[model.tree_.feature[node_id]]
    threshold = model.tree_.threshold[node_id]
    value = sample.iloc[0][feature]
    direction = "LEFT (condition true)" if value <= threshold else "RIGHT (condition false)"
    print(f"Node {node_id}: Is {feature} <= {threshold:.2f}?  "
          f"{feature}={value}  =>  {direction}")

# ---------------------------------------------------------
# 6. Show the decision tree (highlighting is not built-in,
#    but the printed path above tells you exactly which
#    nodes were visited)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
)
plt.title("Decision Tree for Student Performance (50 students)")
plt.tight_layout()
plt.savefig("decision_tree_50.png", dpi=150)
plt.show()