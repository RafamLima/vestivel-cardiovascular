// Wrapper minimalista: NÃO inclui run_classifier.h nem headers DSP.
// Usa apenas o header deste wrapper e o header do modelo TFLite
// que já está dentro da tua biblioteca Stress_inferencing.

#include "ei_run_classifier_stress.h"
#include <tflite-model/tflite_learn_790160_4_compiled.h>
#include <stdlib.h>   // calloc/free

namespace ei {

// allocadores locais (compatíveis com a assinatura do modelo)
static void* stress_calloc(size_t n, size_t size) { return calloc(n, size); }
static void  stress_free(void *p) { free(p); }

EI_IMPULSE_ERROR run_classifier_stress(
    ei::signal_t* signal,
    ei_impulse_result_t* result,
    bool debug
) {
    // 1) Inicializar o modelo TFLite (usa os nossos allocadores)
    if (tflite_learn_790160_4_init(&stress_calloc) != kTfLiteOk) {
        return EI_IMPULSE_TFLITE_ERROR;
    }

    // 2) Obter tensor de entrada
    TfLiteTensor input;
    if (tflite_learn_790160_4_input(0, &input) != kTfLiteOk) {
        tflite_learn_790160_4_reset(&stress_free);
        return EI_IMPULSE_TFLITE_ERROR;
    }

    // 3) Copiar features do signal para o tensor (assumindo layout flat)
    const size_t feature_count = input.bytes / sizeof(float); // para float32
    for (size_t ix = 0; ix < feature_count; ix++) {
        float v = 0.f;
        if (signal->get_data(ix, 1, &v) != 0) {
            tflite_learn_790160_4_reset(&stress_free);
            return EI_IMPULSE_TFLITE_ERROR;
        }
#if EI_CLASSIFIER_TFLITE_INPUT_DATATYPE == EI_CLASSIFIER_DATATYPE_FLOAT32
        reinterpret_cast<float*>(input.data.data)[ix] = v;
#else
        reinterpret_cast<int8_t*>(input.data.data)[ix] =
            static_cast<int8_t>(v / input.params.scale + input.params.zero_point);
#endif
    }

    // 4) Inference
    if (tflite_learn_790160_4_invoke() != kTfLiteOk) {
        tflite_learn_790160_4_reset(&stress_free);
        return EI_IMPULSE_TFLITE_ERROR;
    }

    // 5) Ler saída e preencher result->classification[*]
    TfLiteTensor output;
    if (tflite_learn_790160_4_output(0, &output) != kTfLiteOk) {
        tflite_learn_790160_4_reset(&stress_free);
        return EI_IMPULSE_TFLITE_ERROR;
    }

    // Assume forma [1, N]
    const size_t out_neurons =
        (output.dims && output.dims->size >= 2) ? output.dims->data[1] : 0;

    float sum = 0.f;
    for (size_t i = 0; i < out_neurons; i++) {
#if EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE == EI_CLASSIFIER_DATATYPE_FLOAT32
        float val = reinterpret_cast<float*>(output.data.data)[i];
#else
        float val = (output.data.int8[i] - output.params.zero_point) * output.params.scale;
#endif
        result->classification[i].value = val;
        // A label é preenchida pelo chamador a partir de ei_classifier_inferencing_categories
        sum += val;
    }

    // Normalização simples (se fizer sentido para o teu modelo)
    if (sum > 0.f) {
        for (size_t i = 0; i < out_neurons; i++) {
            result->classification[i].value /= sum;
        }
    }

    // Libertar arena
    tflite_learn_790160_4_reset(&stress_free);
    return EI_IMPULSE_OK;
}

} // namespace ei
