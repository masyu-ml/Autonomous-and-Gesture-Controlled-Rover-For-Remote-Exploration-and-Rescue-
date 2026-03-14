

#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include <WiFi.h>
#include <WiFiUDP.h>

// ------------------------------------------------------------
//  USER CONFIG — only WiFi credentials needed
// ------------------------------------------------------------
#define WIFI_SSID        "***********"
#define WIFI_PASS        "***********"
#define UDP_PORT         5005       // streaming port
#define CMD_PORT         5006       // camera command port
#define DISCOVERY_PORT   5007       // auto IP discovery port
// ------------------------------------------------------------

// WROVER OV2640 pin map
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    21
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      19
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM       5
#define Y2_GPIO_NUM       4
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// ------------------------------------------------------------
//  Tunable constants
// ------------------------------------------------------------
#define CHUNK_SIZE          1400
#define TARGET_FPS          20
#define FRAME_INTERVAL_MS   (1000 / TARGET_FPS)
#define HEALTH_INTERVAL_MS  5000
#define MIN_FRAME_BYTES     1000
#define MAX_FRAME_BYTES     120000
#define HEAP_CRITICAL       20000
#define DISCOVERY_INTERVAL  2000    // broadcast HELLO every 2s until found
// ------------------------------------------------------------

// ------------------------------------------------------------
//  Globals
// ------------------------------------------------------------
WiFiUDP         udp;
WiFiUDP         cmdUdp;
WiFiUDP         discoveryUdp;

static uint32_t frame_id        = 0;
static uint32_t frames_sent     = 0;
static uint32_t frames_dropped  = 0;
static uint32_t last_frame_ms   = 0;
static uint32_t last_health_ms  = 0;
static uint32_t last_hello_ms   = 0;
static uint8_t  pkt_header[16];

// Auto IP discovery state
static IPAddress pc_ip;
static bool      pc_found       = false;

// ------------------------------------------------------------
//  Camera init
// ------------------------------------------------------------
static bool camera_init() {
    pinMode(SIOD_GPIO_NUM, INPUT_PULLUP);
    pinMode(SIOC_GPIO_NUM, INPUT_PULLUP);
    delay(150);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 24000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;
        config.jpeg_quality = 15;
        config.fb_count     = 2;
        config.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[CAM] PSRAM found — VGA / double buffer / GRAB_LATEST");
    } else {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count     = 1;
        config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[CAM] No PSRAM — QVGA / single buffer");
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        Serial.println("[CAM] Sensor handle failed");
        return false;
    }

    // Sensor tuning
    s->set_gain_ctrl(s,   0);
    s->set_agc_gain(s,    5);
    s->set_gainceiling(s, GAINCEILING_4X);
    s->set_aec2(s,        0);
    s->set_ae_level(s,    0);
    s->set_aec_value(s,   300);
    s->set_whitebal(s,    1);
    s->set_awb_gain(s,    1);
    s->set_wb_mode(s,     0);
    s->set_brightness(s,  0);
    s->set_contrast(s,    0);
    s->set_saturation(s,  0);
    s->set_sharpness(s,   0);
    s->set_denoise(s,     1);
    s->set_quality(s,     15);
    s->set_vflip(s,       0);
    s->set_hmirror(s,     0);

    // Drain stale startup frames
    for (int i = 0; i < 3; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(50);
    }

    Serial.println("[CAM] Init OK — stale frames drained");
    return true;
}

// ------------------------------------------------------------
//  WiFi connect
// ------------------------------------------------------------
static bool wifi_connect() {
    Serial.printf("[WIFI] Connecting to: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    for (int i = 0; i < 40; i++) {
        if (WiFi.status() == WL_CONNECTED) {
            WiFi.setTxPower(WIFI_POWER_19_5dBm);
            Serial.printf("[WIFI] Connected → %s | RSSI: %d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
            return true;
        }
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n[WIFI] Failed.");
    return false;
}

// ------------------------------------------------------------
//  Auto IP discovery
//  ESP32 broadcasts "HELLO" until Python replies "HI"
// ------------------------------------------------------------
static void handle_discovery() {
    // Check for incoming HI reply from Python
    int pktSize = discoveryUdp.parsePacket();
    if (pktSize > 0) {
        char buf[16] = {0};
        discoveryUdp.read(buf, sizeof(buf) - 1);
        if (String(buf).startsWith("HI")) {
            pc_ip    = discoveryUdp.remoteIP();
            pc_found = true;
            Serial.printf("[DISCOVERY] Python found at %s ✅\n",
                pc_ip.toString().c_str());
            return;
        }
    }

    // Broadcast HELLO every 2 seconds until Python is found
    if (!pc_found) {
        uint32_t now = millis();
        if (now - last_hello_ms >= DISCOVERY_INTERVAL) {
            last_hello_ms = now;
            discoveryUdp.beginPacket(IPAddress(255,255,255,255), DISCOVERY_PORT);
            discoveryUdp.print("HELLO");
            discoveryUdp.endPacket();
            Serial.println("[DISCOVERY] Broadcasting HELLO...");
        }
    }
}

// ------------------------------------------------------------
//  UDP camera command handler (port 5006)
// ------------------------------------------------------------
static void handle_cmd() {
    int pktSize = cmdUdp.parsePacket();
    if (!pktSize) return;

    char buf[64] = {0};
    int  len     = cmdUdp.read(buf, sizeof(buf) - 1);
    if (len <= 0) return;

    String    cmd = String(buf);
    sensor_t* s   = esp_camera_sensor_get();
    if (!s) return;

    Serial.println("[CMD] " + cmd);

    if      (cmd.startsWith("CMD:EXP:"))       { s->set_exposure_ctrl(s,0); s->set_aec_value(s,   cmd.substring(8).toInt());  }
    else if (cmd.startsWith("CMD:GAIN:"))      { s->set_gain_ctrl(s,0);     s->set_agc_gain(s,    cmd.substring(9).toInt());  }
    else if (cmd.startsWith("CMD:BRIGHTNESS:")){ s->set_brightness(s,       cmd.substring(15).toInt()); }
    else if (cmd.startsWith("CMD:CONTRAST:"))  { s->set_contrast(s,         cmd.substring(13).toInt()); }
    else if (cmd.startsWith("CMD:QUALITY:"))   { s->set_quality(s,          cmd.substring(12).toInt()); }
    else if (cmd.startsWith("CMD:AWB:"))       {
        int val = cmd.substring(8).toInt();
        s->set_whitebal(s, val);
        s->set_awb_gain(s, val);
    }
    else if (cmd == "CMD:AUTO") {
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s,     1);
        s->set_whitebal(s,      1);
        s->set_awb_gain(s,      1);
        Serial.println("[CMD] Auto mode restored");
    }
}

// ------------------------------------------------------------
//  Send one frame as chunked UDP packets
// ------------------------------------------------------------
static bool send_frame(camera_fb_t* fb) {
    if (!fb || fb->len == 0) return false;

    uint32_t total_chunks = (fb->len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (uint32_t i = 0; i < total_chunks; i++) {
        uint32_t offset    = i * CHUNK_SIZE;
        uint32_t chunk_len = min((uint32_t)CHUNK_SIZE,
                                 (uint32_t)(fb->len - offset));

        memcpy(pkt_header,      &frame_id,     4);
        memcpy(pkt_header + 4,  &fb->len,      4);
        memcpy(pkt_header + 8,  &i,            4);
        memcpy(pkt_header + 12, &total_chunks, 4);

        udp.beginPacket(pc_ip, UDP_PORT);
        udp.write(pkt_header, 16);
        udp.write(fb->buf + offset, chunk_len);

        if (udp.endPacket() == 0) {
            Serial.printf("[UDP] Send failed chunk %u\n", i);
            return false;
        }

        if (i % 10 == 0) yield();
        delayMicroseconds(100);
    }
    return true;
}

// ------------------------------------------------------------
//  Health report
// ------------------------------------------------------------
static void print_health() {
    uint32_t free_heap  = esp_get_free_heap_size();
    uint32_t free_psram = psramFound() ? ESP.getFreePsram() : 0;

    Serial.printf(
        "[HEALTH] Sent:%u | Dropped:%u | Heap:%u | PSRAM:%u | PC:%s\n",
        frames_sent, frames_dropped, free_heap, free_psram,
        pc_found ? pc_ip.toString().c_str() : "searching..."
    );

    if (free_heap < HEAP_CRITICAL) {
        Serial.println("[HEALTH] Critical heap — restarting...");
        delay(100);
        ESP.restart();
    }
}

// ------------------------------------------------------------
//  Setup
// ------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("============================================");
    Serial.println("  ESP32 WROVER — UDP Camera Streamer v3");
    Serial.println("============================================");

    if (!camera_init()) {
        Serial.println("[FATAL] Camera init failed.");
        while (true) delay(1000);
    }

    if (!wifi_connect()) {
        Serial.println("[FATAL] WiFi failed.");
        while (true) delay(1000);
    }

    udp.begin(UDP_PORT);
    cmdUdp.begin(CMD_PORT);
    discoveryUdp.begin(DISCOVERY_PORT);

    Serial.printf("[UDP]       Stream port  : %d\n", UDP_PORT);
    Serial.printf("[CMD]       Command port : %d\n", CMD_PORT);
    Serial.printf("[DISCOVERY] Discovery port: %d\n", DISCOVERY_PORT);
    Serial.println("[READY] Waiting for Python...");

    last_frame_ms  = millis();
    last_health_ms = millis();
    last_hello_ms  = millis();
}

// ------------------------------------------------------------
//  Main loop
// ------------------------------------------------------------
void loop() {
    // WiFi watchdog
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WIFI] Lost — reconnecting...");
        WiFi.disconnect();
        delay(500);
        wifi_connect();
        return;
    }

    // Auto IP discovery
    handle_discovery();

    // Handle camera commands
    handle_cmd();

    // Don't stream until Python is found
    if (!pc_found) {
        delay(10);
        return;
    }

    // FPS cap
    uint32_t now = millis();
    if (now - last_frame_ms < FRAME_INTERVAL_MS) {
        delayMicroseconds(100);
        return;
    }
    last_frame_ms = now;

    // Grab latest frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        frames_dropped++;
        delay(10);
        return;
    }

    // Sanity check
    if (fb->len < MIN_FRAME_BYTES || fb->len > MAX_FRAME_BYTES) {
        frames_dropped++;
        esp_camera_fb_return(fb);
        return;
    }

    // Send
    frame_id = (frame_id + 1) % 0xFFFFFFFE;
    bool ok  = send_frame(fb);

    // CRITICAL — always return buffer
    esp_camera_fb_return(fb);

    if (ok) frames_sent++;
    else    frames_dropped++;

    // Health report every 5 seconds
    if (millis() - last_health_ms >= HEALTH_INTERVAL_MS) {
        last_health_ms = millis();
        print_health();
    }
}