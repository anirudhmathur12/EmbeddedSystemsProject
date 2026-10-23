
#ifndef SENSOR_RX_H
#define SENSOR_RX_H

#include "esp_err.h"

#include <stdbool.h>
#include <stdint.h>

// 15 bytes
typedef struct __attribute__((packed)) {
    uint8_t header[2];  // Sync bytes: 0xAA, 0x55
    float ph;          // 4 bytes
    float turbidity;   // 4 bytes
    float temperature; // 4 bytes
    uint8_t checksum;   // 1 byte XOR checksum of the 12 float bytes
} sensor_packet_t;

// cleaned up data structure for in code use
typedef struct {
    float ph;
    float turbidity;
    float temperature;
} sensor_data_t;

esp_err_t sensor_rx_init(void);

bool sensor_rx_read(sensor_data_t* out_data, uint32_t timeout_ms);

#endif // SENSOR_RX_H
