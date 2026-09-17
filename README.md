# ESP32-S3 ML Classifier

A small TensorFlow Lite Micro machine-learning classifier running directly on an ESP32-S3.

## Requirements

- ESP32-S3 development board
- [ESP-IDF](https://github.com/espressif/esp-idf) 6.x
- Python 3

The project uses the ESP-IDF `esp-tflite-micro` component, which will be fetched through the ESP-IDF component manager.

The Python utilities are only required if you want to train or modify the model. They are not required to run inference on the ESP32.
