#include "milk_model_runner.h"
#include "sensor_rx.h"

#include <stdio.h>

void app_main(void) {
    sensor_rx_init();

    sensor_data_t readings;

    if (!sensor_rx_read(&readings, 2000))
        return;

    int xgb_result = run_model_binary_class_xgb(6.65, 25.0, 100.0);
    int rf_result = run_model_multi_class_rf(6.65, 25.0, 100.0);

    printf("XGBoost Result: %d\n", xgb_result);
    printf("Random Forest Result: %d\n", rf_result);
}
