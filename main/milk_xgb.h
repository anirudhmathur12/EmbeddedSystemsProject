#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Inputs: pH, Temperature_C, Turbidity_NTU. Returns 0 on success.
 * output[0] = P(Normal), output[1] = P(Adulterated). No heap allocation.
 * Returns -1 for null pointers and -2 for non-finite inputs.
 */
int milk_xgb_score(const double input[3], double output[2]);
#ifdef __cplusplus
}
#endif
