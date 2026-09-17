# Check if .keras and .tflite models generate the same output

import numpy as np
import tensorflow as tf
import joblib

keras_model = tf.keras.models.load_model("classifier.keras")

interpreter = tf.lite.Interpreter(
    model_path="classifier.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("scaler.pkl")

raw_input = np.array(
    [[0.42, 73.2, 28.5, 614.0]],
    dtype=np.float32
)

model_input = scaler.transform(raw_input).astype(np.float32)

print("Raw input:")
print(raw_input)

print("\nModel input:")
print(model_input)

keras_output = keras_model.predict(
    model_input,
    verbose=0
)

keras_probability = float(keras_output[0][0])

interpreter.set_tensor(
    input_details[0]["index"],
    model_input
)

interpreter.invoke()

tflite_output = interpreter.get_tensor(
    output_details[0]["index"]
)

tflite_probability = float(tflite_output[0][0])

difference = abs(
    keras_probability - tflite_probability
)

print("\nResults")
print("-------")

print(f"Keras:  {keras_probability:.10f}")
print(f"TFLite: {tflite_probability:.10f}")
print(f"Diff:   {difference:.10f}")

print("\nClasses")
print("-------")

keras_class = int(keras_probability >= 0.5)
tflite_class = int(tflite_probability >= 0.5)

print(f"Keras class:  {keras_class}")
print(f"TFLite class: {tflite_class}")

if difference < 1e-5:
    print("\nPASS: Keras and TFLite outputs match.")
else:
    print("\nWARNING: Keras and TFLite outputs differ.")
