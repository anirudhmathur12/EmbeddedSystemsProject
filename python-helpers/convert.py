# Convert .keras to tensorflow

import tensorflow as tf

model = tf.keras.models.load_model("classifier.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("classifier.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Saved classifier.tflite ({len(tflite_model)} bytes)")
