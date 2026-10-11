#include "milk_model_runner.h"

#include <stdio.h>

void app_main(void) {
    int xgb_result = run_model_binary_class_xgb(6.65, 25.0, 100.0);
    int rf_result = run_model_multi_class_rf(6.65, 25.0, 100.0);

    printf("XGBoost Result: %d\n", xgb_result);
    printf("Random Forest Result: %d\n", rf_result);
}
