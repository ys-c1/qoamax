// Differential test matrix: SIMD decoding vs. scalar decoding.
//
// For every cell of the matrix (all encoder efforts x the three materials x the two frame lengths, all configured in
// difftest_config.h) the same encoded slices are decoded:
//   1. by the scalar reference (qoamax_decode_slice, one slice at a time),
//   2. by every compiled SIMD decode kernel that the CPU can run (the qoamax_decode_lanes* variants, one stream per
//      lane, with the stream counts of DIFTEST_STREAM_COUNTS),
//   3. through the public multi-stream API (qoamax_decode_streams, which picks the kernel itself), including chunked
//      decoding with partial slices and interleaved output,
//   4. through the file-format decoder (qoamax_decoder_read / _interleaved), including frame-batch and frame-by-frame
//      paths on a long multi-frame signal.
//
// Every output is compared sample by sample with the scalar reference. The test fails unless the number of differing
// samples and the maximum absolute sample difference are both zero. Per-function coverage of the decoder is collected
// by the gcov build and summarized by test/difftest_coverage.py.
//
// The matrix dimensions come from difftest_config.h: adding an effort, material or frame length there changes nothing
// in this file.
#include "../src/qoamax.c"
#include "difftest_config.h"
#include "difftest_materials.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////
//                                 Utilities                                 //
///////////////////////////////////////////////////////////////////////////////

static int failures;
static uint64_t total_samples_compared, total_mismatches;
static int global_max_diff;

static double difftest_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *xcalloc(size_t size) {
  void *p = calloc(size ? size : 1, 1);
  if (!p) {
    fputs("out of memory\n", stderr);
    exit(EXIT_FAILURE);
  }
  return p;
}

static uint64_t read_u64be(const char *p) {
  uint64_t value = 0;
  for (int i = 0; i < 8; i++) value = (value << 8) | (uint8_t)p[i];
  return value;
}

static void fail(const char *fmt, ...) {
  failures++;
  va_list args;
  va_start(args, fmt);
  fputs("FAIL: ", stderr);
  vfprintf(stderr, fmt, args);
  fputc('\n', stderr);
  va_end(args);
}

//! Compares n samples of `actual` (sample i at actual[i * stride]) with the scalar reference, updating the stats.
static void compare_samples(const char *what, const int16_t *actual, int stride, const int16_t *reference, size_t n,
                            uint64_t *cell_mismatches, int *cell_max_diff) {
  for (size_t i = 0; i < n; i++) {
    int diff = (int)actual[i * stride] - (int)reference[i];
    int abs_diff = diff < 0 ? -diff : diff;
    total_samples_compared++;
    if (abs_diff != 0) {
      (*cell_mismatches)++;
      total_mismatches++;
      if (abs_diff > *cell_max_diff) *cell_max_diff = abs_diff;
      if (abs_diff > global_max_diff) global_max_diff = abs_diff;
      if (total_mismatches <= 10)
        fail("%s: sample %zu: SIMD %d != scalar %d (diff %d)", what, i, actual[i * stride], reference[i], diff);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                          Encoding and parsed QOA file                      //
///////////////////////////////////////////////////////////////////////////////

typedef struct {
  char *data;
  size_t size;
  int channels;
  uint32_t total_samples;
  // Per channel: the slices in host byte order and the LMS state each frame starts with.
  uint64_t **slices;
  qoamax_lms **frame_lms;
  size_t *frame_slice_counts; // slices in each frame
  size_t frame_count;
  size_t slices_per_channel;
} parsed_qoa;

//! Encodes a mono signal into a QOA file in an exactly sized buffer.
static char *encode_qoa(qoamax_encoder_effort effort, int channels, size_t frames, const int16_t *interleaved,
                        size_t *out_size) {
  size_t size = qoamax_encoder_get_encoded_size(channels, frames);
  char *data  = xcalloc(size);
  qoamax_encoder encoder;
  if (qoamax_encoder_init(&encoder, effort, channels, (uint32_t)frames, 48000) != 0)
    fail("qoamax_encoder_init");
  char *pos = data;
  const int16_t *in = interleaved;
  qoamax_encoder_write_interleaved(&encoder, size, &pos, frames * channels, &in, true);
  if ((size_t)(pos - data) != size) fail("encoded %td of %zu bytes", pos - data, size);
  *out_size = size;
  return data;
}

//! Parses an encoded QOA file into its frames, LMS states and host-order slices (the input of every decoder path).
static parsed_qoa parse_qoa(const char *data, size_t size) {
  parsed_qoa q = {0};
  q.data        = xcalloc(size);
  memcpy(q.data, data, size);
  const char *p   = q.data;
  const char *end = q.data + size;

  uint64_t header = read_u64be(p);
  p += 8;
  q.total_samples = (uint32_t)header;
  if ((header >> 32) != QOAMAX_MAGIC) fail("bad QOA magic");
  q.channels = (uint8_t)(read_u64be(p) >> 56); // peek at the first frame header for the channel count

  size_t capacity_frames = 16;
  q.frame_lms            = xcalloc(capacity_frames * sizeof *q.frame_lms);
  q.frame_slice_counts   = xcalloc(capacity_frames * sizeof *q.frame_slice_counts);
  q.slices               = xcalloc(q.channels * sizeof *q.slices);
  size_t slices_capacity = 64;
  for (int ch = 0; ch < q.channels; ch++) q.slices[ch] = xcalloc(slices_capacity * sizeof(uint64_t));

  while (p < end) {
    uint64_t frame_header  = read_u64be(p);
    p += 8;
    int channels           = frame_header >> 56;
    uint16_t frame_samples = frame_header >> 16;
    if (channels != q.channels) fail("channel count changes inside the file");
    size_t slice_count = (frame_samples + 19) / 20;

    if (q.frame_count == capacity_frames) {
      capacity_frames *= 2;
      q.frame_lms = realloc(q.frame_lms, capacity_frames * sizeof *q.frame_lms);
      q.frame_slice_counts = realloc(q.frame_slice_counts, capacity_frames * sizeof *q.frame_slice_counts);
    }
    if (q.slices_per_channel + slice_count > slices_capacity) {
      while (q.slices_per_channel + slice_count > slices_capacity) slices_capacity *= 2;
      for (int ch = 0; ch < q.channels; ch++)
        q.slices[ch] = realloc(q.slices[ch], slices_capacity * sizeof(uint64_t));
    }
    q.frame_lms[q.frame_count]          = xcalloc(channels * sizeof(qoamax_lms));
    q.frame_slice_counts[q.frame_count] = slice_count;
    for (int ch = 0; ch < channels; ch++) {
      uint64_t history = read_u64be(p);
      p += 8;
      uint64_t weights = read_u64be(p);
      p += 8;
      qoamax_lms *lms = &q.frame_lms[q.frame_count][ch];
      for (int i = 0; i < 4; i++) {
        lms->history[i] = (int16_t)(history >> 48);
        lms->weights[i] = (int16_t)(weights >> 48);
        history <<= 16;
        weights <<= 16;
      }
    }

    // The slices of all channels are interleaved in the frame; flatten them into per-channel arrays.
    for (size_t s = 0; s < slice_count; s++)
      for (int ch = 0; ch < channels; ch++) {
        q.slices[ch][q.slices_per_channel + s] = read_u64be(p);
        p += 8;
      }
    q.slices_per_channel += slice_count;
    q.frame_count++;
  }
  return q;
}

static void parsed_qoa_free(parsed_qoa *q) {
  for (int ch = 0; ch < q->channels; ch++) free(q->slices[ch]);
  free(q->slices);
  for (size_t f = 0; f < q->frame_count; f++) free(q->frame_lms[f]);
  free(q->frame_lms);
  free(q->frame_slice_counts);
  free(q->data);
}

//! The scalar ground truth: decodes one channel's slices with qoamax_decode_slice, honoring each frame's stored LMS
//! state and the trailing partial slice.
static int16_t *scalar_reference(const parsed_qoa *q, int channel, size_t samples) {
  int16_t *out = xcalloc(samples * sizeof *out);
  qoamax_lms lms;
  size_t slice_index = 0, sample_index = 0;
  for (size_t f = 0; f < q->frame_count; f++) {
    lms = q->frame_lms[f][channel];
    size_t frame_samples_left = q->frame_slice_counts[f] * 20;
    size_t take               = frame_samples_left;
    if (sample_index + take > samples) take = samples - sample_index;
    size_t slices             = (take + 19) / 20;
    for (size_t s = 0; s < slices; s++) {
      int16_t decoded[20];
      qoamax_decode_slice(&lms, q->slices[channel][slice_index++], decoded);
      size_t n = take >= 20 ? 20 : take;
      memcpy(&out[sample_index], decoded, n * sizeof *out);
      sample_index += n;
      take -= n;
    }
  }
  if (sample_index != samples) fail("scalar reference decoded %zu of %zu samples", sample_index, samples);
  return out;
}

///////////////////////////////////////////////////////////////////////////////
//                     The compiled SIMD kernel variants                      //
///////////////////////////////////////////////////////////////////////////////

typedef struct {
  const char *name;
  int lanes;
  void (*decode)(qoamax__stream *streams, int count, int slice_count);
  bool callable; //!< false for variants needing CPU features this machine lacks (e.g. AVX-512 on an AVX2 CPU)
} kernel_variant;

static bool lanes_supported(int lanes) {
#if defined(__x86_64__) || defined(__i386__)
  if (lanes >= 16) return __builtin_cpu_supports("avx512f");
  if (lanes >= 8) return __builtin_cpu_supports("avx2");
  return true; // SSE2 is part of the x86-64 baseline
#elif defined(__riscv) && defined(__linux__) && __clang_major__ >= 19
  if (!__builtin_cpu_supports("v")) return lanes <= 1;
  if (lanes <= 4) return true;
  unsigned long vlenb;
  __asm__("csrr %0, 0xc22" : "=r"(vlenb));
  return vlenb * 8 >= lanes * 32;
#else
  return lanes <= QOAMAX_LANES; // AArch64 etc.: only the compile-target variant exists
#endif
}

static kernel_variant *variants;
static int variant_count;
static int variant_runs[17]; // indexed by lanes (1..16)

static void build_variants(void) {
  kernel_variant list[4];
  int n = 0;
#ifdef QOAMAX_TARGET_X16
  list[n++] = (kernel_variant){"qoamax_decode_lanes__x16", 16, qoamax_decode_lanes__x16, lanes_supported(16)};
#endif
#ifdef QOAMAX_TARGET_X8
  list[n++] = (kernel_variant){"qoamax_decode_lanes__x8", 8, qoamax_decode_lanes__x8, lanes_supported(8)};
#endif
#ifdef QOAMAX_TARGET_X4
  list[n++] = (kernel_variant){"qoamax_decode_lanes__x4", 4, qoamax_decode_lanes__x4, lanes_supported(4)};
#endif
#if QOAMAX_LANES > 1
  list[n++] = (kernel_variant){"qoamax_decode_lanes", QOAMAX_LANES, qoamax_decode_lanes, lanes_supported(QOAMAX_LANES)};
#endif
  variants      = xcalloc((n ? n : 1) * sizeof *variants);
  variant_count = n;
  memcpy(variants, list, n * sizeof *variants);
}

///////////////////////////////////////////////////////////////////////////////
//                 Level 1: direct kernel vs. scalar, full slices             //
///////////////////////////////////////////////////////////////////////////////

//! Decodes the matrix cell through level 1 (all callable kernels, several stream counts).
static void level1_kernels(const parsed_qoa *q, size_t samples, const int16_t *reference, uint64_t *mismatches,
                           int *max_diff) {
  static const int stream_counts[] = {DIFTEST_STREAM_COUNTS};
  size_t full_slices              = samples / 20;

  for (int vi = 0; vi < variant_count; vi++) {
    if (!variants[vi].callable) continue;
    // Every DIFTEST_STREAM_COUNTS value: qoamax__run_streams groups and refills the streams, exercising the kernel
    // with partially filled lane sets and with more streams than lanes.
    for (size_t ci = 0; ci < sizeof stream_counts / sizeof stream_counts[0]; ci++) {
      int count = stream_counts[ci];
      variant_runs[variants[vi].lanes]++;

      qoamax__stream *streams = xcalloc(count * sizeof *streams);
      int16_t **outputs       = xcalloc(count * sizeof *outputs);
      int stride              = ci % 2 == 0 ? 1 : 2; // also exercise the interleaved-output store path
      for (int l = 0; l < count; l++) {
        outputs[l] = xcalloc((size_t)full_slices * 20 * stride * sizeof *outputs[l]);
        streams[l] = (qoamax__stream){
            .state.lms     = QOAMAX_LMS_INITIAL,
            .slices        = (char *)q->slices[0],
            .slice_step    = 8,
            .big_endian    = false,
            .sample_stride = stride,
            .samples       = outputs[l],
            .samples_size  = full_slices * 20,
            .slice_count   = full_slices,
        };
      }
      qoamax__run_streams(streams, count, variants[vi].decode, variants[vi].lanes,
                          QOAMAX_MAX(variants[vi].lanes / 2, 4), qoamax__decode_stream_scalar);
      for (int l = 0; l < count; l++) {
        compare_samples(variants[vi].name, outputs[l], stride, reference, full_slices * 20, mismatches, max_diff);
        free(outputs[l]);
      }
      free(outputs);
      free(streams);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//        Level 2: public multi-stream API, partial slices and chunks         //
///////////////////////////////////////////////////////////////////////////////

//! qoamax_decode_streams on the cell's slices: several stream counts, both strides, and chunked calls that hand over
//! the slices in random-sized pieces, so partial slices and the scalar tail path are exercised repeatedly. Every
//! result is compared with the scalar reference.
static void level2_public_api(const parsed_qoa *q, size_t samples, const int16_t *reference, uint64_t *mismatches,
                              int *max_diff) {
  static const int stream_counts[] = {DIFTEST_STREAM_COUNTS};
  size_t slice_count               = (samples + 19) / 20;

  for (size_t ci = 0; ci < sizeof stream_counts / sizeof stream_counts[0]; ci++) {
    int count  = stream_counts[ci];
    int stride = ci % 2 == 0 ? 1 : 2;

    // One call with all slices: the common whole-stream path (full SIMD batches plus the scalar tail).
    {
      qoamax_decode_stream *streams = xcalloc(count * sizeof *streams);
      int16_t **outputs             = xcalloc(count * sizeof *outputs);
      for (int l = 0; l < count; l++) {
        outputs[l] = xcalloc(samples * stride * sizeof *outputs[l]);
        streams[l] = qoamax_decode_stream_init(q->slices[0], slice_count, outputs[l], samples, stride);
      }
      qoamax_decode_streams(count, streams);
      for (int l = 0; l < count; l++) {
        compare_samples("qoamax_decode_streams", outputs[l], stride, reference, samples, mismatches, max_diff);
        if (streams[l].samples_size != 0 || streams[l].slices_size != 0)
          fail("one-shot public API stream %d did not finish", l);
        free(outputs[l]);
      }
      free(outputs);
      free(streams);
    }

    // Chunked: expose a random number of slices per call. samples_size always covers all remaining samples, so the
    // library, not the harness, owns the partial-slice handling at the end.
    {
      qoamax_decode_stream *streams = xcalloc(count * sizeof *streams);
      int16_t **outputs             = xcalloc(count * sizeof *outputs);
      for (int l = 0; l < count; l++) {
        outputs[l] = xcalloc(samples * stride * sizeof *outputs[l]);
        streams[l] = qoamax_decode_stream_init(q->slices[0], slice_count, outputs[l], samples, stride);
      }
      uint32_t rng = 0x9E3779B9u ^ (uint32_t)(count * 131 + samples);
      size_t fed   = 0;
      while (fed < slice_count) {
        rng = rng * 1664525u + 1013904223u;
        size_t piece = 1 + rng % 7; // 1..7 slices per call: crosses SIMD batch boundaries (4/8/16 lanes)
        if (piece > slice_count - fed) piece = slice_count - fed;
        for (int l = 0; l < count; l++) streams[l].slices_size = piece;
        qoamax_decode_streams(count, streams);
        fed += piece;
      }
      for (int l = 0; l < count; l++) {
        compare_samples("qoamax_decode_streams (chunked)", outputs[l], stride, reference, samples, mismatches,
                        max_diff);
        if (streams[l].samples_size != 0 || streams[l].slices_size != 0)
          fail("chunked public API stream %d did not finish (%zu samples, %zu slices left)", l,
               streams[l].samples_size, streams[l].slices_size);
        free(outputs[l]);
      }
      free(outputs);
      free(streams);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//           Level 3: file-format decoder on a long multi-frame signal        //
///////////////////////////////////////////////////////////////////////////////

static char *long_file_data;
static size_t long_file_size;
static int16_t *long_reference_flat; // interleaved scalar reference of the long signal

//! Builds the long stereo signal and its scalar reference, and encodes it once (fast effort: this only covers the
//! decoder's file-format and frame-batching code, the efforts themselves are covered by the matrix cells).
static void level3_setup(void) {
  size_t frames   = DIFTEST_LONG_FRAMES;
  int channels    = DIFTEST_LONG_CHANNELS;
  int16_t *stereo = xcalloc(frames * channels * sizeof *stereo);
  for (size_t i = 0; i < frames; i++)
    for (int ch = 0; ch < channels; ch++)
      stereo[i * channels + ch] = difftest_music_sample(i + (size_t)ch * 7919);

  long_file_data = encode_qoa(QOAMAX_ENCODER_EFFORT_ULTRAFAST, channels, frames, stereo, &long_file_size);
  parsed_qoa q   = parse_qoa(long_file_data, long_file_size);
  long_reference_flat = xcalloc(frames * channels * sizeof *long_reference_flat);
  for (int ch = 0; ch < channels; ch++) {
    int16_t *ref = scalar_reference(&q, ch, frames);
    for (size_t i = 0; i < frames; i++) long_reference_flat[i * channels + ch] = ref[i];
    free(ref);
  }
  parsed_qoa_free(&q);
  free(stereo);
}

//! Decodes the long file with both public file decoders, in one call and in small chunks (frame-batch and
//! frame-by-frame paths), and compares with the scalar reference.
static void level3_file_decoder(uint64_t *mismatches, int *max_diff) {
  size_t frames   = DIFTEST_LONG_FRAMES;
  int channels    = DIFTEST_LONG_CHANNELS;

  static const size_t chunks[] = {20, 21, 37, 1000, QOAMAX_FRAME_SAMPLES, QOAMAX_FRAME_SAMPLES + 5,
                                  4 * QOAMAX_FRAME_SAMPLES, SIZE_MAX};
  for (size_t ci = 0; ci < sizeof chunks / sizeof chunks[0]; ci++) {
    // Planar decoder
    {
      char *copy = xcalloc(long_file_size);
      memcpy(copy, long_file_data, long_file_size);
      const char *pos = copy;
      qoamax_decoder decoder;
      qoamax_decoder_query query;
      if (qoamax_decoder_init(&decoder, long_file_size, &pos, &query) != 0) fail("decoder_init (planar)");
      int16_t **planes = xcalloc(channels * sizeof *planes);
      for (int ch = 0; ch < channels; ch++) planes[ch] = xcalloc(frames * sizeof *planes[ch]);
      size_t done = 0;
      while (done < frames) {
        size_t chunk = chunks[ci] == SIZE_MAX ? frames - done : chunks[ci];
        if (chunk > frames - done) chunk = frames - done;
        int16_t *base[QOAMAX_MAX_CHANNEL_COUNT];
        int16_t *out[QOAMAX_MAX_CHANNEL_COUNT];
        for (int ch = 0; ch < channels; ch++) base[ch] = out[ch] = planes[ch] + done;
        qoamax_decoder_read(&decoder, long_file_size - (size_t)(pos - copy), &pos, chunk, out);
        size_t advanced = (size_t)(out[0] - base[0]);
        if (advanced == 0 && chunk < frames - done) {
          // Output room smaller than one slice: the decoder waits for more room, so give it the rest of the file.
          for (int ch = 0; ch < channels; ch++) out[ch] = base[ch];
          qoamax_decoder_read(&decoder, long_file_size - (size_t)(pos - copy), &pos, frames - done, out);
          advanced = (size_t)(out[0] - base[0]);
        }
        if (advanced == 0) {
          fail("planar decoder made no progress at sample %zu", done);
          break;
        }
        done += advanced;
      }
      for (int ch = 0; ch < channels; ch++) {
        int16_t *ref = xcalloc(frames * sizeof *ref);
        for (size_t i = 0; i < frames; i++) ref[i] = long_reference_flat[i * channels + ch];
        compare_samples("qoamax_decoder_read", planes[ch], 1, ref, frames, mismatches, max_diff);
        free(ref);
        free(planes[ch]);
      }
      free(planes);
      free(copy);
    }

    // Interleaved decoder
    {
      char *copy = xcalloc(long_file_size);
      memcpy(copy, long_file_data, long_file_size);
      const char *pos = copy;
      qoamax_decoder decoder;
      qoamax_decoder_query query;
      if (qoamax_decoder_init(&decoder, long_file_size, &pos, &query) != 0) fail("decoder_init (interleaved)");
      int16_t *interleaved = xcalloc(frames * channels * sizeof *interleaved);
      size_t done          = 0;
      while (done < frames) {
        size_t per_channel = chunks[ci] == SIZE_MAX ? frames - done : chunks[ci];
        if (per_channel > frames - done) per_channel = frames - done;
        int16_t *base = interleaved + done * channels;
        int16_t *out  = base;
        qoamax_decoder_read_interleaved(&decoder, long_file_size - (size_t)(pos - copy), &pos,
                                        per_channel * channels, &out);
        size_t advanced = (size_t)(out - base) / channels;
        if (advanced == 0 && per_channel < frames - done) {
          out = base;
          qoamax_decoder_read_interleaved(&decoder, long_file_size - (size_t)(pos - copy), &pos,
                                          (frames - done) * channels, &out);
          advanced = (size_t)(out - base) / channels;
        }
        if (advanced == 0) {
          fail("interleaved decoder made no progress at sample %zu", done);
          break;
        }
        done += advanced;
      }
      compare_samples("qoamax_decoder_read_interleaved", interleaved, 1, long_reference_flat, frames * channels,
                      mismatches, max_diff);
      free(interleaved);
      free(copy);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                   Matrix                                   //
///////////////////////////////////////////////////////////////////////////////

typedef struct {
  char cell_name[128];
  uint64_t samples;
  uint64_t mismatches;
  int max_diff;
} cell_result;

#define DEFINE_EFFORT(name, value) {name, value},
static const struct {
  const char *name;
  qoamax_encoder_effort effort;
} efforts[] = {DIFTEST_EFFORTS(DEFINE_EFFORT)};
#undef DEFINE_EFFORT

#define DEFINE_MATERIAL(name) #name,
static const char *materials[] = {DIFTEST_MATERIALS(DEFINE_MATERIAL)};
#undef DEFINE_MATERIAL

#define DEFINE_LENGTH(value) value,
static const size_t frame_lengths[] = {DIFTEST_FRAME_LENGTHS(DEFINE_LENGTH)};
#undef DEFINE_LENGTH

int main(void) {
  double start = difftest_now();
  build_variants();
  level3_setup();

  size_t effort_count   = sizeof efforts / sizeof efforts[0];
  size_t material_count = sizeof materials / sizeof materials[0];
  size_t length_count   = sizeof frame_lengths / sizeof frame_lengths[0];
  size_t cell_count     = effort_count * material_count * length_count;
  cell_result *cells    = xcalloc(cell_count * sizeof *cells);

  printf("SIMD vs. scalar decoder differential test\n");
  printf("matrix: %zu efforts x %zu materials x %zu lengths = %zu cells\n\n", effort_count, material_count,
         length_count, cell_count);

  size_t cell_index = 0;
  for (size_t e = 0; e < effort_count; e++) {
    for (size_t m = 0; m < material_count; m++) {
      for (size_t li = 0; li < length_count; li++) {
        size_t samples = frame_lengths[li];
        difftest_material material = difftest_make_material(materials[m], samples);
        size_t file_size;
        char *file = encode_qoa(efforts[e].effort, 1, samples, material.samples, &file_size);
        parsed_qoa q = parse_qoa(file, file_size);
        int16_t *reference = scalar_reference(&q, 0, samples);

        cell_result *cell = &cells[cell_index];
        snprintf(cell->cell_name, sizeof cell->cell_name, "%-9s %-5s %4zu samples", efforts[e].name,
                 materials[m], samples);
        cell->samples = samples;

        level1_kernels(&q, samples, reference, &cell->mismatches, &cell->max_diff);
        level2_public_api(&q, samples, reference, &cell->mismatches, &cell->max_diff);

        free(reference);
        parsed_qoa_free(&q);
        free(file);
        free(material.samples);
        cell_index++;
      }
    }
  }

  // The file-format decoder is effort-independent (its long multi-frame signal is built once in level3_setup), so it
  // runs a single time rather than once per effort.
  uint64_t level3_mismatches = 0;
  int level3_max_diff        = 0;
  level3_file_decoder(&level3_mismatches, &level3_max_diff);

  double elapsed = difftest_now() - start;

  // Per-cell report.
  printf("%-32s %10s %12s %12s\n", "cell", "samples", "mismatches", "max |diff|");
  for (size_t c = 0; c < cell_count; c++)
    printf("%-32s %10llu %12llu %12d\n", cells[c].cell_name, (unsigned long long)cells[c].samples,
           (unsigned long long)cells[c].mismatches, cells[c].max_diff);

  printf("\ntotals: %llu samples compared, %llu differing samples, max sample difference %d (required: 0 / 0)\n",
         (unsigned long long)total_samples_compared, (unsigned long long)total_mismatches, global_max_diff);

  printf("\nSIMD kernel invocations by lane count:\n");
  for (int lanes = 1; lanes <= 16; lanes++)
    if (variant_runs[lanes]) printf("  %2d-lane kernels: %d invocations\n", lanes, variant_runs[lanes]);
  printf("compiled variants:\n");
  for (int v = 0; v < variant_count; v++)
    printf("  %-26s %s\n", variants[v].name, variants[v].callable ? "ran" : "not callable on this CPU");

  printf("\nelapsed: %.2f s (budget %.0f s)\n", elapsed, DIFTEST_TIME_BUDGET_SECONDS);

  // Sidecar for the coverage report: which variants the CPU could not run, and the differential stats.
  FILE *side = fopen("difftest_sidecar.json", "w");
  if (side) {
    fprintf(side, "{\n");
    fprintf(side, "  \"elapsed_seconds\": %.3f,\n", elapsed);
    fprintf(side, "  \"time_budget_seconds\": %.1f,\n", DIFTEST_TIME_BUDGET_SECONDS);
    fprintf(side, "  \"samples_compared\": %llu,\n", (unsigned long long)total_samples_compared);
    fprintf(side, "  \"mismatches\": %llu,\n", (unsigned long long)total_mismatches);
    fprintf(side, "  \"max_sample_diff\": %d,\n", global_max_diff);
    fprintf(side, "  \"uncallable_variants\": [");
    bool first = true;
    for (int v = 0; v < variant_count; v++)
      if (!variants[v].callable) {
        fprintf(side, "%s\"%s\"", first ? "" : ", ", variants[v].name);
        first = false;
      }
    fprintf(side, "]\n}\n");
    fclose(side);
  }

  bool over_budget = elapsed > DIFTEST_TIME_BUDGET_SECONDS;
  if (total_mismatches != 0 || global_max_diff != 0)
    fail("%llu samples differ between SIMD and scalar decoding", (unsigned long long)total_mismatches);
  if (over_budget) fail("matrix took %.2f s, over the %.0f s budget", elapsed, DIFTEST_TIME_BUDGET_SECONDS);

  printf("\n%s\n", failures ? "FAILED" : "all differential checks passed");
  free(cells);
  free(long_reference_flat);
  free(long_file_data);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
