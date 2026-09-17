# Test the tflite model

import numpy as np
import tensorflow as tf
import joblib

interpreter = tf.lite.Interpreter(
    model_path="classifier.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("scaler.pkl")

# RAW input
raw = np.array(
    [[0.42, 73.2, 28.5, 614.0]],
    dtype=np.float32
)

x = scaler.transform(raw).astype(np.float32)

print("Raw:")
print(raw)

print("\nNormalized:")
print(x)

interpreter.set_tensor(
    input_details[0]["index"],
    x
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nPrediction:")
print(output)
