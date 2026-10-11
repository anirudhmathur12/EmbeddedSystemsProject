#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Inputs: pH, Temperature_C, Turbidity_NTU. Returns 0 on success.
 * Outputs are class probabilities in model_metadata.json stage2.class_order.
 * No heap allocation. Returns -1 for null pointers and -2 for non-finite inputs.
 */
int milk_rf_score(const double input[3], double output[3]);
#ifdef __cplusplus
}
#endif
