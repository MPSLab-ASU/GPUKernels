#pragma once

#include <cuda_fp16.h>

extern "C" void launch_flash_attn_fwd(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int B, int H, int N, int D, bool causal
);
