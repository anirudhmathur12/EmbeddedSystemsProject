#include "milk_model_config.h"
#include "milk_rf.h"
#include "milk_xgb.h"

int run_model_binary_class_xgb(double ph, double temp_c, double turb_NTU) {
    const double input[3] = {ph, temp_c, turb_NTU};
    double output[2] = {0};

    if (milk_xgb_score(input, output) != 0) {
        return -1;
    }

    return output[1] >= MILK_XGB_ADULTERATED_THRESHOLD ? 1 : 0;
}

int run_model_multi_class_rf(double ph, double temp_c, double turb_NTU) {
    const double input[3] = {ph, temp_c, turb_NTU};
    double output[3] = {0};

    if (milk_rf_score(input, output) != 0) {
        return -1;
    }

    int highest = 0;

    for (int i = 0; i < 3; i++) {
        if (output[i] > output[highest]) {
            highest = i;
        }
    }

    return highest;
}
