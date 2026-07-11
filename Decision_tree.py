import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------
# 1. Create the dataset (50 students) - SYNTHETIC
#    Replace with pd.read_csv("your_file.csv") for real data.
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

FEATURES = ["Study_Hours", "Attendance", "Assignment"]
X = data[FEATURES]
y = data["Result"]

# Keep track of the training range per feature, purely so we can
# warn the user if a new student's values fall far outside what
# the model has ever seen (the model will still predict something,
# but that prediction is an extrapolation and less trustworthy).
FEATURE_RANGES = {f: (X[f].min(), X[f].max()) for f in FEATURES}

# ---------------------------------------------------------
# 2. Train the model
# ---------------------------------------------------------
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
model.fit(X, y)
print(f"Model trained. Training accuracy: {model.score(X, y):.2%}\n")


def explain_prediction(sample_df):
    """Print the exact path the tree took for one sample, and return
    (prediction, probabilities)."""
    prediction = model.predict(sample_df)[0]
    proba = model.predict_proba(sample_df)[0]

    row = sample_df.iloc[0]
    print(f"Student -> " + ", ".join(f"{f}={row[f]}" for f in FEATURES))
    print(f"Result: {'PASS' if prediction == 1 else 'FAIL'}")
    print(f"Confidence: Fail={proba[0]:.1%}, Pass={proba[1]:.1%}")

    # Warn if this student's values are outside anything seen in training
    out_of_range = [
        f for f in FEATURES
        if not (FEATURE_RANGES[f][0] <= row[f] <= FEATURE_RANGES[f][1])
    ]
    if out_of_range:
        print(f"⚠ Note: {', '.join(out_of_range)} is/are outside the range "
              f"seen in training. The model still returns a prediction, "
              f"but it's extrapolating rather than interpolating, so treat "
              f"it with more caution.")

    print("Decision path:")
    feature_names = np.array(FEATURES)
    node_indicator = model.decision_path(sample_df)
    leaf_id = model.apply(sample_df)
    node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]

    for node_id in node_index:
        if leaf_id[0] == node_id:
            print(f"  -> Reached leaf node {node_id}: final prediction = "
                  f"{'PASS' if prediction == 1 else 'FAIL'}")
            continue
        feat = feature_names[model.tree_.feature[node_id]]
        threshold = model.tree_.threshold[node_id]
        value = row[feat]
        direction = "LEFT (condition true)" if value <= threshold else "RIGHT (condition false)"
        print(f"  Node {node_id}: Is {feat} <= {threshold:.2f}?  "
              f"{feat}={value}  =>  {direction}")
    print()
    return prediction, proba


# ---------------------------------------------------------
# 3. DEMO: predict on a few students NOT in the training set
#    at all, to prove the model generalizes rather than
#    memorizing rows.
# ---------------------------------------------------------
print("=" * 60)
print("DEMO: predictions on students that do NOT exist in the")
print("50-row training set (values chosen to not match any row)")
print("=" * 60)

demo_students = pd.DataFrame([
    {"Study_Hours": 6.3, "Attendance": 77, "Assignment": 68},
    {"Study_Hours": 2.1, "Attendance": 40, "Assignment": 35},
    {"Study_Hours": 9.4, "Attendance": 95, "Assignment": 90},
], columns=FEATURES)

for i in range(len(demo_students)):
    sample = demo_students.iloc[[i]]
    # confirm it's genuinely unseen
    match = (X == sample.values).all(axis=1).any()
    print(f"[Row exists in training data? {match}]")
    explain_prediction(sample)

# ---------------------------------------------------------
# 4. Take a NEW student's details from the user and predict
# ---------------------------------------------------------
print("=" * 60)
print("Enter the new student's details to predict Pass/Fail:")
print("=" * 60)
try:
    study_hours_in = float(input("Study Hours: "))
    attendance_in = float(input("Attendance: "))
    assignment_in = float(input("Assignment Marks: "))
except ValueError:
    raise SystemExit("Please enter numeric values only.")

sample = pd.DataFrame(
    [[study_hours_in, attendance_in, assignment_in]], columns=FEATURES
)

print("\n================ PREDICTION ================")
explain_prediction(sample)
print("==============================================")

# ---------------------------------------------------------
# 5. Show the decision tree
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