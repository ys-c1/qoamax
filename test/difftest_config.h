// Configuration of the SIMD vs. scalar decoder differential test matrix (test/difftest.c).
//
// The matrix is fully config-driven: adding an encoder effort, a test material or a frame length means adding one row
// to the X-macros below, nothing in test/difftest.c has to change. The coverage report (test/difftest_coverage.py)
// discovers the compiled decoder functions itself, so new functions need no registration either.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../src/qoamax.h"

// Every encoder effort is exercised: its encoder is the only source of QOA slices available here.
// Add a new effort by adding one row.
#define DIFTEST_EFFORTS(X)                                                                                            \
  X("ultrafast", QOAMAX_ENCODER_EFFORT_ULTRAFAST)                                                                     \
  X("superfast", QOAMAX_ENCODER_EFFORT_SUPERFAST)                                                                     \
  X("veryfast", QOAMAX_ENCODER_EFFORT_VERYFAST)                                                                       \
  X("fast", QOAMAX_ENCODER_EFFORT_FAST)                                                                               \
  X("medium", QOAMAX_ENCODER_EFFORT_MEDIUM)                                                                           \
  X("slow", QOAMAX_ENCODER_EFFORT_SLOW)                                                                               \
  X("veryslow", QOAMAX_ENCODER_EFFORT_VERYSLOW)                                                                       \
  X("superslow", QOAMAX_ENCODER_EFFORT_SUPERSLOW)                                                                     \
  X("placebo", QOAMAX_ENCODER_EFFORT_PLACEBO)

// The three material classes. Add one by adding a row and a matching generator in difftest_materials.h.
#define DIFTEST_MATERIALS(X) X(sine) X(noise) X(music)

// The two frame (stream) lengths in samples per channel. They differ from each other in the number of full slices
// and in the size of the trailing partial slice, so that both the SIMD kernel (full slices) and the scalar tail path
// (partial slices) are covered. They stay short because the PLACEBO effort encodes only a few hundred samples per
// second: the whole matrix must finish in under DIFTEST_TIME_BUDGET_SECONDS.
#define DIFTEST_FRAME_LENGTHS(X) X(640) X(240)

// Matrix cells: efforts * materials * frame lengths.
#define DIFTEST_TIME_BUDGET_SECONDS 60.0

// Number of independent streams handed to the SIMD kernels. The values cross every lane-fill threshold of the
// compiled variants (at least half the lanes, and four streams: see qoamax__run_streams), up to the widest variant.
#define DIFTEST_STREAM_COUNTS 1, 2, 3, 4, 5, 7, 8, 9, 16

// The long signal of the file-format tests spans several QOA frames, so that qoamax_decoder_read decodes several
// frames at once in SIMD lanes (see QOAMAX_DECODER_STREAMS), including its frame-by-frame fallback path.
#define DIFTEST_LONG_FRAMES (9 * QOAMAX_FRAME_SAMPLES + 257)
#define DIFTEST_LONG_CHANNELS 2

// One generated test material: deterministic, so every cell is reproducible.
typedef struct {
  const char *name;
  int16_t *samples; // mono, `length` samples
  size_t length;
} difftest_material;
