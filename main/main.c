#include "milk_model_runner.h"
#include "sensor_rx.h"

#include "esp_log.h"

#include <stdio.h>

void app_main(void) {
    sensor_rx_init();
    sensor_data_t readings = {0};

    while(1) {
    	if (!sensor_rx_read(&readings, 1000000)){
	    printf("RX read failed\n");
    	    return;
    	}
    	printf("%f, %f, %f\n", readings.ph, readings.turbidity, readings.temperature);

    	int xgb_result = run_model_binary_class_xgb(readings.ph, readings.temperature, readings.turbidity);
    	int rf_result = run_model_multi_class_rf(readings.ph, readings.temperature, readings.turbidity);

    	printf("XGBoost Result: %d\n", xgb_result);
    	printf("Random Forest Result: %d\n", rf_result);
    }
}
