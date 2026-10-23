#include "sensor_rx.h"

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"

#define TAG "SENSOR_RX"

#define SENSOR_UART_PORT UART_NUM_2
#define SENSOR_UART_BAUD 115200
#define SENSOR_RX_PIN GPIO_NUM_1
#define SENSOR_TX_PIN GPIO_NUM_2
#define RX_BUF_SIZE 256

#define SYNC_BYTE_1 0xAA
#define SYNC_BYTE_2 0x55
#define PAYLOAD_SIZE 12 // 3 floats * 4 bytes

esp_err_t sensor_rx_init(void) {
    uart_config_t uart_config = {
        .baud_rate = SENSOR_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t err = uart_driver_install(SENSOR_UART_PORT, RX_BUF_SIZE * 2, 0, 0, NULL, 0);
    if (err != ESP_OK)
        return err;

    err = uart_param_config(SENSOR_UART_PORT, &uart_config);
    if (err != ESP_OK)
        return err;

    return uart_set_pin(SENSOR_UART_PORT, SENSOR_TX_PIN, SENSOR_RX_PIN, UART_PIN_NO_CHANGE,
                        UART_PIN_NO_CHANGE);
}

static uint8_t calculate_checksum(const uint8_t* payload, size_t len) {
    uint8_t xor_sum = 0;
    for (size_t i = 0; i < len; i++) {
        xor_sum ^= payload[i];
    }
    return xor_sum;
}

bool sensor_rx_read(sensor_data_t* out_data, uint32_t timeout_ms) {
    if (!out_data)
        return false;

    int64_t start_time = esp_timer_get_time();
    int64_t timeout_us = (int64_t)timeout_ms * 1000;
    uint8_t byte = 0;

    // look for sync header: 0xAA 0x55
    while ((esp_timer_get_time() - start_time) < timeout_us) {
        if (uart_read_bytes(SENSOR_UART_PORT, &byte, 1, pdMS_TO_TICKS(20)) == 1) {
            if (byte == SYNC_BYTE_1) {
                if (uart_read_bytes(SENSOR_UART_PORT, &byte, 1, pdMS_TO_TICKS(20)) == 1) {
                    if (byte == SYNC_BYTE_2) {
                        uint8_t raw_payload[PAYLOAD_SIZE + 1];
                        int len = uart_read_bytes(SENSOR_UART_PORT, raw_payload,
                                                  sizeof(raw_payload), pdMS_TO_TICKS(100));

                        if (len == sizeof(raw_payload)) {
                            uint8_t expected_crc = calculate_checksum(raw_payload, PAYLOAD_SIZE);
                            uint8_t received_crc = raw_payload[PAYLOAD_SIZE];

                            if (expected_crc == received_crc) {
                                __builtin_memcpy(out_data, raw_payload,
                                                 PAYLOAD_SIZE); // copy into struct directly
                                return true;
                            } else {
                                ESP_LOGW(TAG, "Checksum mismatch! Expected 0x%02X, got 0x%02X",
                                         expected_crc, received_crc);
                            }
                        }
                    }
                }
            }
        }
    }

    return false; // Timed out waiting for a valid packet
}
