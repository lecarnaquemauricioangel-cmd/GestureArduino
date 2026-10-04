/*
  led_gesture.ino
  ---------------
  Controla un LED en D13 mediante comandos serie enviados por el frontend.

  Protocolo:
    '1' → LED ON  (mano_abierta / Paper)
    '0' → LED OFF (puno_cerrado / Rock)

  Circuito:
    D13 → ánodo LED → cátodo → resistencia 220 Ω → GND
*/

const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(LED_PIN, HIGH);
    } else if (cmd == '0') {
      digitalWrite(LED_PIN, LOW);
    }
    // Flush remaining bytes
    while (Serial.available() > 0) Serial.read();
  }
}
