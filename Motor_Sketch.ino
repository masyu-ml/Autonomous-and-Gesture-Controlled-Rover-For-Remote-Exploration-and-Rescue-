#include <WiFi.h>
#include <WebSocketsServer.h>
#include <Arduino.h>
#include <ESP32Servo.h>

// ==== WiFi credentials ====
const char* ssid = "Rover_AP";
const char* password = "rover1129";

// ==== Motor driver pins (TB6612FNG) ====
#define AIN1 13
#define AIN2 14
#define BIN1 16
#define BIN2 17
#define PWMA 18
#define PWMB 19
#define STBY 21

// ==== Servo pin ====
#define SERVO_PIN 22
Servo clawServo;

// ==== WebSocket server ====
WebSocketsServer webSocket = WebSocketsServer(81);

// ==== Rover state ====
int speedPWM = 130;       // default motor speed (0–255)
const int MIN_SPEED = 100; // minimum speed to avoid stall
const int MAX_SPEED = 150; // maximum safe PWM value
int clawOpen = 30;         // servo angle for open claw
int clawClose = 108;       // servo angle for closed claw
String currentMode = "MANUAL";  // MANUAL / GESTURE / AUTO

// ----------------- Motor functions -----------------
void forward(int speed) {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
  analogWrite(PWMA, speed);
  analogWrite(PWMB, speed);
}

void backward(int speed) {
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  analogWrite(PWMA, speed);
  analogWrite(PWMB, speed);
}

void turnLeft(int speed) {
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
  analogWrite(PWMA, speed);
  analogWrite(PWMB, speed);
}

void turnRight(int speed) {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  analogWrite(PWMA, speed);
  analogWrite(PWMB, speed);
}

void stopMotors() {
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

// ----------------- Servo (claw) functions -----------------
void openClaw() { clawServo.write(clawOpen); }
void closeClaw() { clawServo.write(clawClose); }

// ----------------- WebSocket command handler -----------------
void handleCommand(String cmd) {
  cmd.trim();
  Serial.println("CMD: " + cmd);

  // === Mode switching ===
  if (cmd == "MODE_MANUAL") { currentMode = "MANUAL"; Serial.println("Switched to MANUAL mode"); return; }
  else if (cmd == "MODE_GESTURE") { currentMode = "GESTURE"; Serial.println("Switched to GESTURE mode"); return; }
  else if (cmd == "MODE_AUTO") { currentMode = "AUTO"; Serial.println("Switched to AUTONOMOUS mode"); return; }

  // === Rover motion ===
  if (cmd == "F") forward(speedPWM);
  else if (cmd == "B") backward(speedPWM);
  else if (cmd == "L") turnLeft(speedPWM);
  else if (cmd == "R") turnRight(speedPWM);
  else if (cmd == "STOP") stopMotors();
  else if (cmd == "OPEN") openClaw();
  else if (cmd == "CLOSE") closeClaw();
  
  // === Speed control ===
  else if (cmd.startsWith("SPD")) {
    int val = cmd.substring(3).toInt();
    if (val < MIN_SPEED) val = MIN_SPEED;
    if (val > MAX_SPEED) val = MAX_SPEED;
    speedPWM = val;
    Serial.println("Speed set to " + String(speedPWM));

    // Apply immediately if motors are active
    analogWrite(PWMA, speedPWM);
    analogWrite(PWMB, speedPWM);
  }

  else {
    Serial.println("Unknown cmd: " + cmd);
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if (type == WStype_TEXT) {
    String msg = String((char*)payload);
    handleCommand(msg);
  }
}

// ----------------- Setup -----------------
void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);

  // Servo setup
  clawServo.attach(SERVO_PIN);
  openClaw();

  // WiFi setup
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP: " + WiFi.localIP().toString());

  // WebSocket setup
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket server started on port 81");
}

// ----------------- Loop -----------------
void loop() {
  webSocket.loop();
}
