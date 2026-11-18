// Deterministic generators for the three material classes of the differential test matrix.
//
// All three are generated from fixed constants only (no external files, no platform-dependent state), so the matrix
// is bit-for-bit reproducible on every machine. The generators deliberately include clipping and near-silence, which
// exercise the decoder's int16 saturation.
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "difftest_config.h"

static inline int16_t difftest_clamp_s16(int32_t value) {
  return value < INT16_MIN ? INT16_MIN : value > INT16_MAX ? INT16_MAX : (int16_t)value;
}

//! A small deterministic PRNG (xorshift32), seeded per material so different channels/positions stay independent.
static inline uint32_t difftest_xorshift(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

//! Sine material: two sine waves (one near the Nyquist frequency) plus a quiet high-frequency component, with enough
//! amplitude at the peaks to clip occasionally. Smooth and predictable, so any single-sample mismatch stands out.
static inline int16_t difftest_sine_sample(size_t i) {
  const double pi = 3.14159265358979323846;
  double phase    = 2.0 * pi * (220.0 * i / 48000.0) + 0.25 * sin(2.0 * pi * i / 977.0);
  double sample   = 26000.0 * sin(phase) + 12000.0 * sin(2.0 * pi * 0.31 * i) + 900.0 * sin(2.0 * pi * 0.47 * i);
  return difftest_clamp_s16((int32_t)lround(sample));
}

//! Noise material: full-scale white noise from the PRNG, whose residuals drive every scalefactor of the encoder.
static inline int16_t difftest_noise_sample(size_t i) {
  uint32_t state = 0x12345678u ^ (uint32_t)(i * 2654435761u);
  state          = difftest_xorshift(state);
  state          = difftest_xorshift(state);
  return (int16_t)(state & 0xFFFF);
}

//! "Real music" material: a deterministic stand-in for a musical signal — a few harmonized tones with a slow amplitude
//! envelope, quiet wide-band noise between the notes, and occasional loud transients (the attacks of the notes).
static inline int16_t difftest_music_sample(size_t i) {
  const double pi = 3.14159265358979323846;
  size_t note     = (i / 2400) % 8;
  double local    = (double)(i % 2400);
  // Frequencies of a major scale (Hz), each note with its second and third harmonic.
  static const double notes[] = {261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25};
  double f        = notes[note];
  double envelope = exp(-3.0 * local / 2400.0); // plucked-note decay
  double tone     = envelope * (sin(2 * pi * f * i / 48000.0) + 0.5 * sin(2 * pi * 2 * f * i / 48000.0) +
                                0.25 * sin(2 * pi * 3 * f * i / 48000.0));
  double sample   = 24000.0 * tone / 1.75;

  uint32_t state = 0xABCDEF01u ^ (uint32_t)(i * 2246822519u);
  state          = difftest_xorshift(state);
  sample += (double)((int32_t)(state & 0xFFFF) - 32768) * 0.02; // room noise
  if (i % 2400 < 40) sample += 18000.0 * exp(-(double)(i % 2400) / 10.0); // note attack
  return difftest_clamp_s16((int32_t)lround(sample));
}

static inline difftest_material difftest_make_material(const char *name, size_t length) {
  int16_t *samples = malloc(length * sizeof *samples);
  for (size_t i = 0; i < length; i++) {
    if (strcmp(name, "sine") == 0)
      samples[i] = difftest_sine_sample(i);
    else if (strcmp(name, "noise") == 0)
      samples[i] = difftest_noise_sample(i);
    else
      samples[i] = difftest_music_sample(i);
  }
  return (difftest_material){.name = name, .samples = samples, .length = length};
}
