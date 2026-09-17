# Train the model on dataset.csv
# Generates .keras

import numpy as np
import pandas as pd
import tensorflow as tf

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("dataset.csv")

FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
]

X = df[FEATURES].values.astype(np.float32)
y = df["target"].values.astype(np.float32)

print(f"Dataset: {len(df)} samples")
print(f"Class distribution:")
print(df["target"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

joblib.dump(scaler, "scaler.pkl")

print("\nNormalization parameters:")

for i, feature in enumerate(FEATURES):
    print(
        f"{feature}: "
        f"mean={scaler.mean_[i]:.6f}, "
        f"std={scaler.scale_[i]:.6f}"
    )

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

probabilities = model.predict(X_test).flatten()
predictions = (probabilities >= 0.5).astype(int)

print("\nResults")
print("-------")

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(f"F1:        {f1_score(y_test, predictions):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nExample predictions")
print("-------------------")

for i in range(10):
    print(
        f"Input: {X[i]}  "
        f"probability={probabilities[i]:.4f}  "
        f"predicted={predictions[i]}  "
        f"actual={int(y_test[i])}"
    )

model.save("classifier.keras")

print("\nSaved model to classifier.keras")
