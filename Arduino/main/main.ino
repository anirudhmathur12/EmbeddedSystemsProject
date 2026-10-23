#include <SoftwareSerial.h>
#include <DallasTemperature.h>
#include <OneWire.h>
#define ONE_WIRE_BUS 2
#define PH_OFFSET 0.00
OneWire oneWire(ONE_WIRE_BUS);
const int turbidityPin = A1;
DallasTemperature sensors(&oneWire);


typedef struct __attribute__((packed)) {
  uint8_t header[2];
  double ph;
  double turbidity;
  double temperature;
  uint8_t checksum;
} sensor_packet_t;

// Create a software serial port on pins 9 (RX) and 10 (TX)
// Connect Arduino Pin 9 to ESP32 TX
// Connect Arduino Pin 10 to ESP32 RX
SoftwareSerial mySerial(9, 10);

static uint8_t calculate_checksum(const uint8_t* payload, size_t len) {
    uint8_t xor_sum = 0;
    for (size_t i = 0; i < len; i++) {
        xor_sum ^= payload[i];
    }
    return xor_sum;
}

double temp(){
  double temp = 0;
  double oldtemp = 100;
  while(abs(temp - oldtemp)>0.01){}
    oldtemp = temp;
    sensors.requestTemperatures();
    temp = sensors.getTempCByIndex(0);
  }
  return temp;


}

double turb(){
  long total = 0;
  const int samples = 20;

  double turbidityValue = 0;
  double oldturbidityValue = 5000;

  while(abs(oldturbidityValue-turbidityValue)>0.01){
    oldturbidityValue = turbidityValue;
    // Average several readings to reduce noise
    for (int i = 0; i < samples; i++) {
      total += analogRead(turbidityPin);
      delay(10);
    }

    double rawValue = total / (float)samples;
    double voltage = rawValue * (5.0 / 1023.0);

    turbidityValue = map(rawValue, 780, 0, 0, 5000);
  }
  return turbidityValue;
}

double pH(){
  const int SensorPin = A0;
  const double VRef = 5.00;
  double oldpH = 0.0;
  double pH = 7.00;

  // Measured voltage in your pH 7.00 buffer
  const double V_pH7 = 2.5;

  // Approximate sensitivity; requires calibration
  const double SLOPE = -0.18;

  int buf[10]; 

  // Take 10 samples
  while(abs(oldpH - pH)>0.01){
    oldpH = pH;
    for (int i = 0; i < 10; i++) {
      buf[i] = analogRead(SensorPin);
      delay(10);
    }

    // Sort samples
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 10; j++) {
          if (buf[i] > buf[j]) {
            int temp = buf[i];
            buf[i] = buf[j];
            buf[j] = temp;
          }
        }
      }

      // Average the middle 6 samples
      long totalADC = 0;

      for (int i = 2; i < 8; i++) {
        totalADC += buf[i];
      }

      double avgADC = totalADC / 6.0;
      double voltage = avgADC * VRef / 1023.0;

      // Calibrate the offset using pH 7 buffer
      pH = 7.00 + (V_pH7 - voltage) / SLOPE;
      pH += PH_OFFSET;
  }

    return pH;
}
void setup() {
  // Start the USB Serial Monitor for debugging
  Serial.begin(115200);

  // Start the Software Serial port for the ESP32
  mySerial.begin(115200);
  sensors.begin();
  sensors.setResolution(12);
  pinMode(13, OUTPUT);

  Serial.println("Arduino ready and listening...");
}

void loop() {
  // Send data to the ESP32

  delay(1000);  // Wait 1 second between messages

  double _temp = temp();
  double _pH = pH();
  double _turb = turb();

  sensor_packet_t _packet = (sensor_packet_t) {
    .header = {0xAA, 0x55},
    .ph = _pH,
    .turbidity = _turb,
    .temperature = _temp,
    .checksum = 0,
  };

  uint8_t checksum = calculate_checksum((uint8_t *)(&_packet) + 2, 12);
  _packet.checksum = checksum;

  Serial.println(_pH);
  Serial.println(_turb);
  Serial.println(_temp);
  Serial.println("cheeks");

  mySerial.print("Hello from Apte");

  mySerial.write((uint8_t *)&_packet, sizeof(sensor_packet_t));
}
