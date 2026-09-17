#include <stdio.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "ESP_TEST";

// ESP-IDF embeds classifier.tflite and exposes these symbols. 
extern const uint8_t model_start[] asm(
    "_binary_classifier_tflite_start"
);

extern const uint8_t model_end[] asm(
    "_binary_classifier_tflite_end"
);


/*
 * These values must be the SAME mean/std values used by
 * StandardScaler during training.
 *
 * From the current synthetic dataset generation/training setup.
 */
static const float feature_mean[4] = {
    0.500521287f,
    49.6915423f,
    30.0336302f,
    503.142790f
};

static const float feature_std[4] = {
    0.150398338f,
    19.9147157f,
    7.86067772f,
    198.240093f
};


// Tensor memory arena
constexpr size_t TENSOR_ARENA_SIZE = 64 * 1024;

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];


static const float raw_input[4] = {
    0.42f,
    73.2f,
    28.5f,
    614.0f
};


static void run_inference()
{
    const tflite::Model *model =
        tflite::GetModel(model_start);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(
            TAG,
            "Model schema version %d != supported version %d",
            model->version(),
            TFLITE_SCHEMA_VERSION
        );

        return;
    }

    ESP_LOGI(
        TAG,
        "Model size: %u bytes",
        (unsigned)(
            model_end - model_start
        )
    );


    static tflite::MicroMutableOpResolver<3> resolver;

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to add FullyConnected");
        return;
    }

    if (resolver.AddRelu() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to add Relu");
        return;
    }

    if (resolver.AddLogistic() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to add Logistic");
        return;
    }


    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );


    TfLiteStatus status = interpreter.AllocateTensors();

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }


    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);


    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Could not get input/output tensors");
        return;
    }


    // Verify model input
    if (input->type != kTfLiteFloat32) {
        ESP_LOGE(
            TAG,
            "Expected float32 input, got type %d",
            input->type
        );

        return;
    }

    if (input->dims->size != 2 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != 4) {

        ESP_LOGE(TAG, "Unexpected input shape");

        return;
    }


    // Verify model output
    if (output->type != kTfLiteFloat32) {
        ESP_LOGE(
            TAG,
            "Expected float32 output, got type %d",
            output->type
        );

        return;
    }


    // Apply standard scaler
    ESP_LOGI(TAG, "Raw input:");

    for (int i = 0; i < 4; ++i) {
        ESP_LOGI(
            TAG,
            "  feature_%d = %.6f",
            i + 1,
            raw_input[i]
        );
    }


    ESP_LOGI(TAG, "Normalized input:");

    for (int i = 0; i < 4; ++i) {

        float normalized =
            (raw_input[i] - feature_mean[i])
            / feature_std[i];

        input->data.f[i] = normalized;

        ESP_LOGI(
            TAG,
            "  feature_%d = %.6f",
            i + 1,
            normalized
        );
    }


    ESP_LOGI(TAG, "Running inference...");

    int64_t start = esp_timer_get_time();

    status = interpreter.Invoke();

    int64_t end = esp_timer_get_time();

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return;
    }


    float probability = output->data.f[0];

    int predicted_class =
        probability >= 0.5f ? 1 : 0;


    ESP_LOGI(
        TAG,
        "Inference time: %lld us",
        end - start
    );

    ESP_LOGI(
        TAG,
        "Probability: %.8f",
        probability
    );

    ESP_LOGI(
        TAG,
        "Predicted class: %d",
        predicted_class
    );
}


extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-S3 ML test");

    run_inference();

    // Keep application alive so serial monitor stays connected
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
