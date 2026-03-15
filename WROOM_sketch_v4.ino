#include <WiFi.h>
#include <WebSocketsServer.h>
#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// ==== WiFi Credentials ====
const char* ssid = "Rover_AP";
const char* password = "rover1129";

// ==== Motor Driver Pins ====
#define AIN1 13
#define AIN2 14
#define BIN1 16
#define BIN2 17
#define PWMA 18
#define PWMB 19
#define STBY 21
#define SERVO_PIN 22
#define SDA_PIN 25
#define SCL_PIN 26

// ==== Objects ====
Servo clawServo;
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
WebSocketsServer webSocket = WebSocketsServer(81);

// ==== Global State (Volatile for Dual-Core Safety) ====
volatile int speedPWM = 130;
volatile int targetAngle = 30; 
volatile bool isAutoMode = false; // Booleans are 100x faster than String comparisons
volatile bool sensorFound = false;
volatile uint16_t currentDistance = 9999;

TaskHandle_t Task_ToF;

// ----------------- CORE 0: TOF SENSOR THREAD -----------------
// This runs in the background. It can freeze and block all it wants 
// without EVER delaying your motor/gesture commands!
void ToFSensorTask(void * pvParameters) {
  for(;;) {
    if (sensorFound && isAutoMode) {
      VL53L0X_RangingMeasurementData_t measure;
      lox.rangingTest(&measure, false); // <--- Slow blocking call trapped on Core 0
      
      if (measure.RangeStatus != 4) {
        currentDistance = measure.RangeMilliMeter;
      } else {
        currentDistance = 9999;
      }
    }
    // Rest the sensor for 80ms (roughly 12 frames a second)
    vTaskDelay(80 / portTICK_PERIOD_MS); 
  }
}

// ----------------- MOTOR FUNCTIONS -----------------
void moveForward(int s) { 
  digitalWrite(STBY, HIGH); 
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH); 
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW); 
  analogWrite(PWMA, s); analogWrite(PWMB, s); 
}

void moveBackward(int s) { 
  digitalWrite(STBY, HIGH); 
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW); 
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH); 
  analogWrite(PWMA, s); analogWrite(PWMB, s); 
}

void moveLeft(int s) { 
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  analogWrite(PWMA, s); analogWrite(PWMB, s);
}

void moveRight(int s) { 
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
  analogWrite(PWMA, s); analogWrite(PWMB, s);
}

void stopMotors() { 
  digitalWrite(STBY, LOW); 
  analogWrite(PWMA, 0); analogWrite(PWMB, 0); 
}

// ----------------- WebSocket Events (CORE 1) -----------------
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if (type == WStype_TEXT) {
    String cmd = String((char*)payload);
    cmd.trim();
    
    if (cmd == "MODE_AUTO") isAutoMode = true;
    else if (cmd == "MODE_MANUAL") {
      isAutoMode = false;
      stopMotors(); // Safety halt when switching modes
    }
    
    // Servo Commands
    else if (cmd == "OPEN") {
      targetAngle = 30;
      clawServo.write(targetAngle);
    }
    else if (cmd == "CLOSE") {
      targetAngle = 108;
      clawServo.write(targetAngle);
    }
    
    // Motor Commands
    else if (cmd == "F")    moveForward(speedPWM);
    else if (cmd == "B")    moveBackward(speedPWM);
    else if (cmd == "L")    moveLeft(speedPWM);
    else if (cmd == "R")    moveRight(speedPWM);
    else if (cmd == "STOP") stopMotors();
    
    else if (cmd.startsWith("SPD")) speedPWM = cmd.substring(3).toInt();
  }
}

// ----------------- SETUP -----------------
void setup() {
  Serial.begin(115200);

  // 1. ALLOCATE TIMERS FIRST
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // 2. SETUP SERVO
  clawServo.setPeriodHertz(50);
  clawServo.attach(SERVO_PIN, 500, 2400);
  clawServo.write(targetAngle); 

  // 3. SETUP MOTORS
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT); pinMode(PWMB, OUTPUT);
  pinMode(STBY, OUTPUT);
  stopMotors();

  // 4. SETUP WIFI (LATENCY OPTIMIZED)
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // <--- CRITICAL: Forces Wi-Fi radio to stay awake 100% of the time!
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  
  // 5. SETUP SENSORS & WEBSOCKETS
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Wire.begin(SDA_PIN, SCL_PIN);
  if (lox.begin()) sensorFound = true;

  // 6. LAUNCH BACKGROUND SENSOR THREAD ON CORE 0
  xTaskCreatePinnedToCore(
    ToFSensorTask,   // Task function
    "ToF_Task",      // Name
    4096,            // Stack size
    NULL,            // Parameters
    1,               // Priority
    &Task_ToF,       // Handle
    0                // Core 0 (Background Core)
  );
}

// ----------------- MAIN LOOP (CORE 1) -----------------
void loop() {
  // WebSockets will now process INSTANTLY because this loop never freezes
  webSocket.loop();
  
  // Handle Auto-Mode Telemetry & Servo Logic safely on Core 1
  static uint32_t lastBroadcast = 0;
  if (isAutoMode && millis() - lastBroadcast > 100) {
    lastBroadcast = millis();
    
    // Read the volatile variable updated by Core 0
    uint16_t dist = currentDistance; 
    
    // Broadcast to Python
    webSocket.broadcastTXT("DIST:" + String(dist));
    
    // Auto-Claw Response
    if (dist < 100 && targetAngle != 108) {
      targetAngle = 108; // Close
      clawServo.write(targetAngle);
    } 
    else if (dist >= 100 && targetAngle != 30) {
      targetAngle = 30;  // Open
      clawServo.write(targetAngle);
    }
  }
}
