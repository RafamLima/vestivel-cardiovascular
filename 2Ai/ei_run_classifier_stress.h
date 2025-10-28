#pragma once

// Precisamos das definições de EI_IMPULSE_ERROR, ei_impulse_result_t e signal_t
#include <edge-impulse-sdk/classifier/ei_classifier_types.h>

namespace ei {
    EI_IMPULSE_ERROR run_classifier_stress(
        ei::signal_t *signal,
        ei_impulse_result_t *result,
        bool debug
    );
}
