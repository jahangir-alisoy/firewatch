/*
  FireWatch — Arduino Uno R3 node

  Works in two modes:
    LINKED  : the PC sends the fused AI+sensor level (0..10). LEDs show max(PC, local).
    LOCAL   : no message from PC for 3 s -> the board decides alone from its own sensors.

  WIRING (see the table in the chat reply)
    D2..D11  -> 10 LEDs (each through 330 Ω to GND). D2 = LED 1 ... D11 = LED 10
    D12      -> buzzer (via NPN transistor, or a 3-pin active buzzer module's I/O)
    D13      -> on-board LED = PC link indicator
    A0       -> MQ-2 module AO
    A1       -> LM35 Vout
    A2       -> IR flame module AO (optional, set USE_FLAME 0 if not connected)
    A3       -> push button to GND (mute 60 s)

  Serial 115200
    Board -> PC : D,smokeRaw,smokeBase,temp_x10,flameRaw,local,shown,muted,warm
    PC -> Board : L<0..10>   fused level
                  M          mute buzzer 60 s
*/

#define USE_FLAME 1
#define BUZZER_ACTIVE_LOW 0   // set 1 if your buzzer module beeps when the pin is LOW

const uint8_t LED_PINS[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t PIN_BUZZER = 12;
const uint8_t PIN_LINK   = 13;
const uint8_t PIN_SMOKE  = A0;
const uint8_t PIN_TEMP   = A1;
const uint8_t PIN_FLAME  = A2;
const uint8_t PIN_BUTTON = A3;

// ---- tuning --------------------------------------------------------------
const unsigned long WARMUP_MS    = 20000;  // MQ-2 heater warm-up
const unsigned long LINK_TIMEOUT = 3000;
const unsigned long MUTE_MS      = 60000;
const unsigned long REPORT_MS    = 250;

const int   SMOKE_DELTA_MIN = 60;    // raw counts above baseline -> level 1
const int   SMOKE_DELTA_MAX = 400;   // -> level 10
const float TEMP_MIN        = 45.0;  // °C -> level 1
const float TEMP_MAX        = 70.0;  // °C -> level 10
const float ROR_LIMIT       = 8.0;   // °C rise per minute -> at least level 6
const int   FLAME_MIN       = 250;   // intensity = 1023 - raw
const int   FLAME_MAX       = 850;

const uint8_t LEVEL_WARN  = 4;
const uint8_t LEVEL_ALARM = 7;

// ---- state ---------------------------------------------------------------
float smokeBase = 0;
float tempC = 0;
int   smokeRaw = 0, flameRaw = 1023;
uint8_t localLevel = 0, pcLevel = 0, shownLevel = 0;

unsigned long lastPcMsg = 0, muteUntil = 0, lastReport = 0;
bool linked = false, warm = true, everLinked = false;

float tempRing[7];            // one sample every 10 s -> 60 s window
uint8_t ringIdx = 0, ringFill = 0;
unsigned long lastRingSample = 0;

char rxBuf[16];
uint8_t rxLen = 0;

// ---- helpers -------------------------------------------------------------
int readAvg(uint8_t pin) {
  long sum = 0;
  for (uint8_t i = 0; i < 8; i++) sum += analogRead(pin);
  return sum / 8;
}

uint8_t scale(float v, float lo, float hi) {
  if (v < lo) return 0;
  if (v >= hi) return 10;
  return 1 + (uint8_t)((v - lo) * 9.0 / (hi - lo));
}

bool muted() { return millis() < muteUntil; }

void handleCommand(const char *cmd) {
  if (cmd[0] == 'L') {
    int v = atoi(cmd + 1);
    pcLevel = constrain(v, 0, 10);
    lastPcMsg = millis();
    everLinked = true;
  } else if (cmd[0] == 'M') {
    muteUntil = millis() + MUTE_MS;
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen) { rxBuf[rxLen] = 0; handleCommand(rxBuf); rxLen = 0; }
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    }
  }
}

void readButton() {
  static bool last = HIGH;
  static unsigned long changedAt = 0;
  bool now = digitalRead(PIN_BUTTON);
  if (now != last && millis() - changedAt > 40) {
    changedAt = millis();
    last = now;
    if (now == LOW) muteUntil = muted() ? 0 : millis() + MUTE_MS;   // press toggles mute
  }
}

void readSensors() {
  smokeRaw = readAvg(PIN_SMOKE);
  float t = readAvg(PIN_TEMP) * 500.0 / 1023.0;               // LM35: 10 mV/°C at 5 V ref
  tempC = (tempC == 0) ? t : tempC * 0.8 + t * 0.2;
#if USE_FLAME
  flameRaw = readAvg(PIN_FLAME);
#endif

  if (warm) {
    smokeBase = (smokeBase == 0) ? smokeRaw : smokeBase * 0.95 + smokeRaw * 0.05;
    if (millis() > WARMUP_MS) warm = false;
  } else if (smokeRaw - smokeBase < SMOKE_DELTA_MIN / 2) {
    smokeBase += (smokeRaw - smokeBase) * 0.001;                // slow drift compensation
  }

  if (millis() - lastRingSample >= 10000 || ringFill == 0) {
    lastRingSample = millis();
    tempRing[ringIdx] = tempC;
    ringIdx = (ringIdx + 1) % 7;
    if (ringFill < 7) ringFill++;
  }
}

uint8_t computeLocal() {
  if (warm) return 0;

  uint8_t smokeL = scale(smokeRaw - smokeBase, SMOKE_DELTA_MIN, SMOKE_DELTA_MAX);
  uint8_t tempL  = scale(tempC, TEMP_MIN, TEMP_MAX);
  if (ringFill == 7) {
    float rise = tempC - tempRing[ringIdx];                     // oldest sample = 60 s ago
    if (rise >= ROR_LIMIT && tempL < 6) tempL = 6;
  }
  uint8_t flameL = 0;
#if USE_FLAME
  flameL = scale(1023 - flameRaw, FLAME_MIN, FLAME_MAX);
#endif

  uint8_t lvl = max(smokeL, max(tempL, flameL));
  uint8_t agreeing = (smokeL >= 3) + (tempL >= 3) + (flameL >= 3);
  if (agreeing >= 2) lvl = min(10, lvl + 2);                    // two sensors agree -> escalate
  return lvl;
}

void drawLeds() {
  if (warm && shownLevel == 0) {                                // chase animation while warming up
    uint8_t pos = (millis() / 120) % 10;
    for (uint8_t i = 0; i < 10; i++) digitalWrite(LED_PINS[i], i == pos);
    return;
  }
  for (uint8_t i = 0; i < 10; i++) digitalWrite(LED_PINS[i], i < shownLevel);
}

void driveBuzzer() {
  unsigned long t = millis();
  bool on = false;
  if (!muted()) {
    if (shownLevel >= LEVEL_ALARM)     on = (t / 250) % 2;      // 2 Hz beeping
    else if (shownLevel >= LEVEL_WARN) on = (t % 3000) < 60;    // short chirp every 3 s
  }
  digitalWrite(PIN_BUZZER, on ^ BUZZER_ACTIVE_LOW);
}

void report() {
  Serial.print("D,");
  Serial.print(smokeRaw);              Serial.print(',');
  Serial.print((int)smokeBase);        Serial.print(',');
  Serial.print((int)(tempC * 10));     Serial.print(',');
  Serial.print(flameRaw);              Serial.print(',');
  Serial.print(localLevel);            Serial.print(',');
  Serial.print(shownLevel);            Serial.print(',');
  Serial.print(muted() ? 1 : 0);       Serial.print(',');
  Serial.println(warm ? 1 : 0);
}

// ---- main ----------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < 10; i++) pinMode(LED_PINS[i], OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LINK, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
}

void loop() {
  readSerial();
  readButton();
  readSensors();

  linked = everLinked && (millis() - lastPcMsg < LINK_TIMEOUT);
  localLevel = computeLocal();
  shownLevel = linked ? max(pcLevel, localLevel) : localLevel;

  drawLeds();
  driveBuzzer();
  digitalWrite(PIN_LINK, warm ? (millis() / 500) % 2 : linked);

  if (millis() - lastReport >= REPORT_MS) {
    lastReport = millis();
    report();
  }
}
