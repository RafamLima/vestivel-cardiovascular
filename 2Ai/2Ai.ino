// === ESP32 Dual-core: ECG 250 Hz + envio em bloco 1 Hz (PLOTTER: ECG apenas) ===
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ===== [EI] Edge Impulse: incluir a tua biblioteca exportada =====
#include <An_lise_BPM_inferencing.h>                       // cabeçalho da tua lib (ZIP)
#include <edge-impulse-sdk/classifier/ei_run_classifier.h> // run_classifier()

#include <Stress_inferencing.h>     // isto puxa a lib e o include path dela
#include "ei_run_classifier_stress.h"  // agora o header está no próprio sketch

// [STRESS AI] ---- parâmetros do teu impulso (4 eixos @10 Hz, janela 15 s = 150 amostras)
#ifndef EI_STRESS_LABEL_COUNT
  #define EI_STRESS_LABEL_COUNT 2   // normal, stress (ajusta se o teu projeto tiver outro nº de classes)
#endif
static const int STRESS_FS_HZ = 10;                 // 10 Hz
static const int STRESS_WIN_MS = 15000;             // 15 s
static const int STRESS_SAMPLES = STRESS_FS_HZ * (STRESS_WIN_MS / 1000); // 150
static const int STRESS_AXES = 4;                   // pulso, respiracao, gsr, temperatura
static const int STRESS_TOTAL = STRESS_SAMPLES * STRESS_AXES;            // 600

#define ECG_PIN    36
#define PIEZO_PIN  35
#define GSR_PIN    34
#define MLX_SDA    17
#define MLX_SCL    16
#define MPU_SDA    23
#define MPU_SCL    19
#define LED_PIN     2

// --- Config ---
#define FS_HZ                250
#define SAMPLE_PERIOD_MS     4
#define POST_PERIOD_MS       1000
#define TEMP_PERIOD_MS       100
#define USE_BATCH            1     // envia array de ECG (1 bloco/segundo)

// ===== MODO PLOTTER: mostrar só ECG no Serial Plotter =====
#define PLOTTER_ECG_ONLY     0     // <--- se quiseres ver probabilidades no Monitor Série, mete 0

// --- Debug flags (auto-ajustados pelo modo plotter) ---
#ifndef DEBUG_PLOT
#define DEBUG_PLOT 0
#endif
#ifndef DEBUG_HTTP
#define DEBUG_HTTP 0
#endif

#if PLOTTER_ECG_ONLY
  #undef DEBUG_PLOT
  #define DEBUG_PLOT 1     // imprime apenas o ECG filtrado (uma coluna)
  #undef DEBUG_HTTP
  #define DEBUG_HTTP 0     // silencia prints de POST
#endif

// ===== [NET] interruptor para não enviar nada =====
#ifndef DISABLE_NETWORK
#define DISABLE_NETWORK 0   // 1 = não usa Wi-Fi/POST; 0 = usa como dantes
#endif

// --- Wi-Fi / server ---
const char* WIFI_SSID = "Vodafone-C6E2E1 2.4G";
const char* WIFI_PASS = "lima1970";
const char* URL_SINGLE = "http://192.168.1.84/website/website/api/guardar.php";
const char* URL_BATCH  = "http://192.168.1.84/website/website/api/guardar_batch.php";

// ======================= [ADICIONADO] =======================
// URL do endpoint para guardar AI de BPM
const char* URL_AI_BPM = "http://192.168.1.84/website/website/api/guardar_ai.php";
// ======================= [ADICIONADO – Stress API] =======================
const char* URL_AI_STRESS = "http://192.168.1.84/website/website/api/guardar_ai_stress.php";
// ========================================================================

// Mensagem para fila das decisões AI BPM
typedef struct {
  float p_bradi, p_normal, p_taqui;
  char  label[16];
  uint32_t ts_ms;
} ai_bpm_msg_t;

static QueueHandle_t qAI_BPM;   // fila para decisões da AI de BPM

// ======================= [ADICIONADO – Stress API] =======================
typedef struct {
  float p_normal, p_stress;
  char  label[12];       // "normal" ou "stress"
  uint32_t ts_ms;
} ai_stress_msg_t;

static QueueHandle_t qAI_STRESS;   // fila para decisões da AI de Stress
// ========================================================================

// --- Sensores ---
Adafruit_MLX90614 mlx;
Adafruit_MPU6050  mpu;

// --- Filtros (visual apenas) ---
const float FC_HP = 1.2f, FC_LP = 22.0f, VIEW_GAIN = 8.0f;
float a_hp, a_lp, dc = 0.f, lp = 0.f;
int q0=0, q1=0;

// --- Estado lento (atualizado no sender) ---
volatile float last_temp = NAN;
volatile int   last_piezo = 0, last_gsr = 0;
volatile float last_ax=0, last_ay=0, last_az=0;

// --- Queue de amostras ECG ---
typedef struct { uint16_t raw; } ecg_t;
static QueueHandle_t qECG;

// --- Buffers para envio ---
static uint16_t ecg_block[FS_HZ];   // 1 segundo

// --- Util LEDs ---
static inline void ledOK(){ digitalWrite(LED_PIN, HIGH); delay(6); digitalWrite(LED_PIN, LOW); }
static inline void ledERR(){ for(int i=0;i<2;i++){ digitalWrite(LED_PIN,HIGH); delay(35); digitalWrite(LED_PIN,LOW); delay(70);} }

// --- Wi-Fi ---
static void wifiEnsure() {
  #if DISABLE_NETWORK
    return; // rede desativada
  #else
  if (WiFi.status()==WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0=millis();
  while(WiFi.status()!=WL_CONNECTED && millis()-t0<12000){ delay(200); }
  #endif
}

/* =======================================================================
   [EI - MODO ANTIGO]  <<<<< ÚNICA ALTERAÇÃO >>>>>
   Alimenta o classificador com APENAS 35 amostras e imprime probabilidades.
   ======================================================================= */
// Buffer do EI: 35 pontos (sem normalização, como no código antigo)
static float  ei_buf[EI_CLASSIFIER_RAW_SAMPLE_COUNT];  // 35
static size_t ei_fill = 0;

// Probabilidades + label
static float  ai_bradi = NAN, ai_normal = NAN, ai_taqui = NAN;
static char   ai_label[24] = "";

// Callback para o signal_t
static int ei_signal_get_data(size_t offset, size_t length, float *out_ptr){
  if (offset + length > EI_CLASSIFIER_RAW_SAMPLE_COUNT) return EIDSP_OUT_OF_MEM;
  for (size_t i=0; i<length; i++) out_ptr[i] = ei_buf[offset + i];
  return EIDSP_OK;
}

// Alimenta até perfazer 35 amostras
static inline void feed_ai_bpm_with_block(const uint16_t *src, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (ei_fill >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) break; // já temos 35
    ei_buf[ei_fill++] = (float)src[i];                    // mantém escala original
  }
}

// Corre a IA e imprime probabilidades
static void run_ai_bpm_now() {
  if (ei_fill < EI_CLASSIFIER_RAW_SAMPLE_COUNT) return;

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data     = &ei_signal_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) { ei_fill = 0; return; }

  ai_bradi = ai_normal = ai_taqui = NAN;
  float best_val = -1.f; int best_idx = -1;

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    const char* lbl = result.classification[ix].label;
    float val = result.classification[ix].value;

    if      (strcasecmp(lbl, "bradicardia") == 0) ai_bradi  = val;
    else if (strcasecmp(lbl, "normal")      == 0) ai_normal = val;
    else if (strcasecmp(lbl, "taquicardia") == 0) ai_taqui  = val;

    if (val > best_val) { best_val = val; best_idx = (int)ix; }
  }

  if (best_idx >= 0) {
    size_t k=0; const char* s = result.classification[best_idx].label;
    for (; s[k] && k < sizeof(ai_label)-1; k++) ai_label[k] = (char)tolower((unsigned char)s[k]);
    ai_label[k] = '\0';
  } else {
    ai_label[0] = '\0';
  }

  #if !PLOTTER_ECG_ONLY
    Serial.printf("AI: %s  [ Bradicardia=%.3f | Normal=%.3f | Taquicardia=%.3f ]\n",
      ai_label[0] ? ai_label : "NA",
      isfinite(ai_bradi)?ai_bradi:0.f,
      isfinite(ai_normal)?ai_normal:0.f,
      isfinite(ai_taqui)?ai_taqui:0.f
    );
  #endif

  // ======================= [ADICIONADO] =======================
  // Enfileira a decisão para a task de envio
  ai_bpm_msg_t msg;
  msg.p_bradi  = isfinite(ai_bradi)  ? ai_bradi  : 0.f;
  msg.p_normal = isfinite(ai_normal) ? ai_normal : 0.f;
  msg.p_taqui  = isfinite(ai_taqui)  ? ai_taqui  : 0.f;
  snprintf(msg.label, sizeof(msg.label), "%s", ai_label[0] ? ai_label : "na");
  msg.ts_ms = millis();
  if (qAI_BPM) xQueueSend(qAI_BPM, &msg, 0);
  // ============================================================

  ei_fill = 0; // prepara próxima janela (35 pontos)
}
/* ======================= FIM DA ALTERAÇÃO ======================= */

/* ======================= [STRESS AI] PARTE NOVA ======================= */
// Buffer (intercalado por amostra): [pulso, resp, gsr, temp] * 150 = 600
static float stress_buf[STRESS_TOTAL];
static int   stress_count = 0;

// Downsample do pulso (ECG 250 Hz -> 10 Hz): média dos samples em cada 100 ms
static uint32_t ecg_sum_100ms = 0;
static uint16_t ecg_cnt_100ms = 0;

// Callback para o signal_t da IA de stress
static int stress_signal_get_data(size_t offset, size_t length, float *out_ptr){
  if (offset + length > (size_t)STRESS_TOTAL) return EIDSP_OUT_OF_MEM;
  for (size_t i=0; i<length; i++) out_ptr[i] = stress_buf[offset + i];
  return EIDSP_OK;
}

// Corre a IA de stress e imprime decisão (sem enviar nada)
static void run_ai_stress_now() {
  if (stress_count < STRESS_SAMPLES) return; // ainda não temos 15 s

  ei::signal_t signal_s;
  signal_s.total_length = STRESS_TOTAL;
  signal_s.get_data     = &stress_signal_get_data;

  ei_impulse_result_t result_s = { 0 };
  EI_IMPULSE_ERROR err_s = ei::run_classifier_stress(&signal_s, &result_s, false);

  // Se houve erro na inferência, mostra e aborta esta janela
  if (err_s != EI_IMPULSE_OK) {
    Serial.printf("STRESS AI ERRO: %d\n", (int)err_s);
    stress_count = 0;
    return;
  }

  // ====== ALTERAÇÃO ÚNICA AQUI: usar índices fixos (0=normal, 1=stress) ======
  float p_normal = result_s.classification[0].value;
  float p_stress = result_s.classification[1].value;
  const char* best = (p_normal >= p_stress) ? "normal" : "stress";
  // ============================================================================

  Serial.printf("STRESS AI: %s  [ normal=%.3f | stress=%.3f ]\n",
                best,
                isfinite(p_normal)?p_normal:0.f,
                isfinite(p_stress)?p_stress:0.f);

  // ------- [ADICIONADO] Enfileirar decisão para task dedicada -------
  ai_stress_msg_t sm;
  sm.p_normal = isfinite(p_normal) ? p_normal : 0.f;
  sm.p_stress = isfinite(p_stress) ? p_stress : 0.f;
  snprintf(sm.label, sizeof(sm.label), "%s", best);   // "normal" / "stress"
  sm.ts_ms = millis();
  if (qAI_STRESS) xQueueSend(qAI_STRESS, &sm, 0);
  // ------------------------------------------------------------------

  // preparar próxima janela de 15 s
  stress_count = 0;
}
/* =================== FIM [STRESS AI] PARTE NOVA =================== */

// ========== [NET] ALTERAÇÃO: HTTPClient único com timeouts ==========
#if !DISABLE_NETWORK
static HTTPClient g_http;          // reutilizado
static bool g_http_inited = false; // só configuramos uma vez
#endif
// ====================================================================

// ========== TASK: AQUISIÇÃO (Core 0, alta prioridade) ==========
void TaskAcquire(void*){
  TickType_t lastWake = xTaskGetTickCount();
  for(;;){
    uint16_t raw = (uint16_t)analogRead(ECG_PIN);

    // envia crua para queue
    ecg_t s{raw};
    xQueueSend(qECG, &s, 0);

    // --- PLOT: imprime apenas ECG filtrado (1 valor por linha) ---
    #if DEBUG_PLOT
      float x   = (float)raw;
      dc += a_hp * (x - dc);
      float ac = x - dc;
      lp += a_lp * (ac - lp);
      float y = lp;
      float smooth = (y + q0 + q1) / 3.0f;
      q1 = q0; q0 = (int)y;

      // IMPORTANTE: só este println vai para a Serial -> uma coluna = 1 gráfico
      Serial.println(smooth * VIEW_GAIN, 3);
    #endif

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
  }
}

// ========== TASK: ENVIO + [EI] INFERÊNCIA (Core 1) ==========
void TaskSend(void*){
  uint32_t nextTempMs = 0, nextPostMs = 0;
  uint16_t n = 0;

  for(;;){
    ecg_t s;
    if (xQueueReceive(qECG, &s, pdMS_TO_TICKS(4)) == pdTRUE) {
      if (n < FS_HZ) ecg_block[n++] = s.raw;

      // [STRESS AI] acumular pulso para média a cada 100 ms (downsample para 10 Hz)
      ecg_sum_100ms += s.raw;
      ecg_cnt_100ms++;
    }

    // ===== [sensores lentos] =====
    uint32_t now = millis();
    if (now >= nextTempMs){
      last_temp  = mlx.readObjectTempC();
      last_piezo = analogRead(PIEZO_PIN);   // respiracao
      last_gsr   = analogRead(GSR_PIN);     // gsr
      sensors_event_t a, g, t_mpu;
      mpu.getEvent(&a, &g, &t_mpu);
      last_ax=a.acceleration.x; last_ay=a.acceleration.y; last_az=a.acceleration.z;
      nextTempMs = now + TEMP_PERIOD_MS;

      // [STRESS AI] gerar 1 amostra a 10 Hz (100 ms): pulso_ds, respiracao, gsr, temperatura
      float pulso_ds = (ecg_cnt_100ms > 0) ? (float)ecg_sum_100ms / (float)ecg_cnt_100ms : 0.0f;
      ecg_sum_100ms = 0; ecg_cnt_100ms = 0;

      if (stress_count < STRESS_SAMPLES) {
        // *** EIXO-MAIOR: todos os 150 samples por eixo
        int k = stress_count; // 0..149
        stress_buf[0 * STRESS_SAMPLES + k] = pulso_ds;           // pulso (ADC 0–4095)
        stress_buf[1 * STRESS_SAMPLES + k] = (float)last_piezo;  // respiração (ADC 0–4095)
        stress_buf[2 * STRESS_SAMPLES + k] = (float)last_gsr;    // GSR (ADC 0–4095)
        stress_buf[3 * STRESS_SAMPLES + k] = isnan(last_temp) ? 36.0f : last_temp; // °C
        stress_count++;
      }

      // Quando tivermos 15 s (150 amostras), corre a IA de stress e imprime
      if (stress_count >= STRESS_SAMPLES) {
        run_ai_stress_now();
      }
    }

    // ===== [Ciclo de 1 s] =====
    now = millis();   // <- correção: reutiliza a variável, não redeclara
    if ((now >= nextPostMs || n >= FS_HZ) && n > 0){
      wifiEnsure();

      // ===== IA BPM (modo antigo: 35 pontos do último segundo) =====
      feed_ai_bpm_with_block(ecg_block, n);   // mete até 35 pontos
      run_ai_bpm_now();                       // classifica e imprime

      // ===== [POST / Wi-Fi] — mantido mas DESLIGADO se DISABLE_NETWORK=1 =====
      #if !DISABLE_NETWORK
      if (WiFi.status()==WL_CONNECTED){

        // [NET] ALTERAÇÃO: inicializar HTTPClient 1x com timeouts curtos
        if (!g_http_inited) {
          g_http.setConnectTimeout(1500); // ms
          g_http.setTimeout(1500);        // ms (leitura)
          g_http.setReuse(true);          // tenta manter ligação viva
          g_http_inited = true;
        }

        #if USE_BATCH
          g_http.begin(URL_BATCH);
          g_http.addHeader("Content-Type","application/json");
          String json = "{";
          json += "\"paciente_id\":\"444134\",";   // <<<<<< ALTERADO
          json += "\"temperatura\":" + String(isnan(last_temp)?0.0:last_temp,2) + ",";
          json += "\"gsr\":" + String(last_gsr) + ",";
          json += "\"respiracao\":" + String(last_piezo) + ",";
          json += "\"ax\":" + String(last_ax,3) + ",";
          json += "\"ay\":" + String(last_ay,3) + ",";
          json += "\"az\":" + String(last_az,3) + ",";
          json += "\"ecg\":[";
          for(uint16_t i=0;i<n;i++){ json += String(ecg_block[i]); if(i+1<n) json += ","; }
          json += "]}";
          int code = g_http.POST(json);
          String resp = g_http.getString();
        #else
          g_http.begin(URL_SINGLE);
          g_http.addHeader("Content-Type","application/json");
          String json = "{";
          json += "\"paciente_id\":\"444134\",";   // <<<<<< ALTERADO
          json += "\"pulso\":" + String(n? ecg_block[n-1] : 0) + ",";
          json += "\"respiracao\":" + String(last_piezo) + ",";
          json += "\"gsr\":" + String(last_gsr) + ",";
          json += "\"temperatura\":" + String(isnan(last_temp)?0.0:last_temp,2) + ",";
          json += "\"ax\":" + String(last_ax,3) + ",";
          json += "\"ay\":" + String(last_ay,3) + ",";
          json += "\"az\":" + String(last_az,3) + "}";
          int code = g_http.POST(json);
          String resp = g_http.getString();
        #endif

        #if DEBUG_HTTP
          Serial.printf("\n[POST] code=%d json_len=%d\n", code, (int)json.length());
          if (resp.length()) Serial.println(resp);
        #endif

        if (code>0 && code<400) ledOK(); else ledERR();
        g_http.end(); // fecha a request; objeto continua a ser reutilizado
      } else {
        ledERR();
        #if DEBUG_HTTP
          Serial.println("[POST] Wi-Fi OFF");
        #endif
      }
      #endif // !DISABLE_NETWORK

      // preparar próximo segundo
      n = 0;
      nextPostMs = now + POST_PERIOD_MS;
    }
  }
}

// ======================= [ADICIONADO] =======================
// Task dedicada: envia decisões da AI de BPM para guardar_ai.php
void TaskSendAIBPM(void*) {
  #if !DISABLE_NETWORK
  static HTTPClient http;
  static bool http_inited = false;
  #endif

  char last_label[16] = "";
  uint32_t lastSentMs = 0;
  const uint32_t MIN_PERIOD_MS = 800; // anti-flood opcional

  for(;;){
    ai_bpm_msg_t m;
    if (xQueueReceive(qAI_BPM, &m, pdMS_TO_TICKS(1000)) == pdTRUE) {

      // opcional: só envia se mudou de classe ou passou tempo mínimo
      if (strcmp(last_label, m.label) == 0 && (millis()-lastSentMs) < MIN_PERIOD_MS) {
        continue;
      }

      wifiEnsure();

      #if !DISABLE_NETWORK
      if (WiFi.status()==WL_CONNECTED) {
        if (!http_inited) {
          http.setConnectTimeout(1500);
          http.setTimeout(1500);
          http.setReuse(true);
          http_inited = true;
        }

        String json = "{";
        json += "\"paciente_id\":\"444134\",";   // <<<<<< ALTERADO
        json += "\"p_bradi\":"  + String(m.p_bradi, 5)  + ",";
        json += "\"p_normal\":" + String(m.p_normal,5)  + ",";
        json += "\"p_taqui\":"  + String(m.p_taqui, 5)  + ",";
        json += "\"label\":\""  + String(m.label) + "\"";
        json += "}";

        http.begin(URL_AI_BPM);
        http.addHeader("Content-Type","application/json");
        int code = http.POST(json);
        String resp = http.getString();
        http.end();

        if (code>0 && code<400) { ledOK(); lastSentMs = millis(); strncpy(last_label, m.label, sizeof(last_label)); }
        else { ledERR(); }
      } else {
        ledERR();
      }
      #endif
    }
  }
}
// ============================================================

// ======================= [ADICIONADO] =======================
// Task dedicada: envia decisões da AI de STRESS para guardar_ai_stress.php
void TaskSendAIStress(void*) {
  #if !DISABLE_NETWORK
  static HTTPClient http;
  static bool http_inited = false;
  #endif

  char last_label[12] = "";
  uint32_t lastSentMs = 0;
  const uint32_t MIN_PERIOD_MS = 1000; // evita flood se o estado não muda

  for(;;){
    ai_stress_msg_t m;
    if (xQueueReceive(qAI_STRESS, &m, pdMS_TO_TICKS(1500)) == pdTRUE) {

      // só envia se mudou o label ou passou tempo mínimo
      if (strcmp(last_label, m.label) == 0 && (millis() - lastSentMs) < MIN_PERIOD_MS) {
        continue;
      }

      wifiEnsure();

      #if !DISABLE_NETWORK
      if (WiFi.status() == WL_CONNECTED) {
        if (!http_inited) {
          http.setConnectTimeout(1500);
          http.setTimeout(1500);
          http.setReuse(true);
          http_inited = true;
        }

        String json = "{";
        json += "\"paciente_id\":\"444134\",";         // mesmo paciente_id que usas no resto
        json += "\"p_normal\":" + String(m.p_normal, 5) + ",";
        json += "\"p_stress\":" + String(m.p_stress, 5) + ",";
        json += "\"label\":\"" + String(m.label) + "\"";
        json += "}";

        http.begin(URL_AI_STRESS);
        http.addHeader("Content-Type","application/json");
        int code = http.POST(json);
        String resp = http.getString();
        http.end();

        if (code > 0 && code < 400) {
          ledOK();
          lastSentMs = millis();
          strncpy(last_label, m.label, sizeof(last_label));
        } else {
          ledERR();
        }
      } else {
        ledERR();
      }
      #endif
    }
  }
}
// ============================================================

// ======================= SETUP =======================
void setup(){
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT); digitalWrite(LED_PIN, LOW);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN,   ADC_11db);
  analogSetPinAttenuation(PIEZO_PIN, ADC_11db);
  analogSetPinAttenuation(GSR_PIN,   ADC_11db);

  Wire.begin(MLX_SDA, MLX_SCL);
  Wire1.begin(MPU_SDA, MPU_SCL);

  // Evitar texto no Plotter: só mostra avisos se NÃO estivermos em modo plotter
  #if !PLOTTER_ECG_ONLY
    if(!mlx.begin(MLX90614_I2CADDR, &Wire)) Serial.println("MLX90614 nao encontrado");
    if(!mpu.begin(0x68, &Wire1)) Serial.println("MPU6050 nao encontrado");
  #else
    mlx.begin(MLX90614_I2CADDR, &Wire);
    mpu.begin(0x68, &Wire1);
  #endif

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  float Ts = 1.0f/FS_HZ, pi = 3.14159265f;
  a_hp = Ts / ((1.0f/(2.0f*pi*FC_HP)) + Ts);
  a_lp = Ts / ((1.0f/(2.0f*pi*FC_LP)) + Ts);

  wifiEnsure();

  const uint16_t QUEUE_LEN = FS_HZ * 3; // ~3 s de margem
  qECG = xQueueCreate(QUEUE_LEN, sizeof(ecg_t));

  // [ADICIONADO] fila e task de envio da AI BPM
  qAI_BPM = xQueueCreate(16, sizeof(ai_bpm_msg_t));
  xTaskCreatePinnedToCore(TaskSendAIBPM, "SendAIBPM", 6144, nullptr, 1, nullptr, 1);

  // [ADICIONADO] fila e task de envio da AI de STRESS
  qAI_STRESS = xQueueCreate(12, sizeof(ai_stress_msg_t));
  xTaskCreatePinnedToCore(TaskSendAIStress, "SendAIStress", 6144, nullptr, 1, nullptr, 1);

  // (sem DDA/binning — modo antigo)
  // Inicialização de estados da IA já está feita nos estáticos

  xTaskCreatePinnedToCore(TaskAcquire, "Acquire250Hz", 4096, nullptr, 3, nullptr, 0);
  xTaskCreatePinnedToCore(TaskSend,    "Send1Hz",      8192, nullptr, 1, nullptr, 1);
}

void loop(){ /* tudo em tasks */ }
