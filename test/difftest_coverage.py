#!/usr/bin/env python3
"""Summarize the per-function decoder coverage of the SIMD/scalar differential test.

Input: the gcov JSON intermediate file produced from the instrumented test binary, plus the sidecar JSON written by
the test itself (difftest_sidecar.json: differential stats, runtime and the SIMD variants the CPU cannot run).

The report is precise to the function. Decoder functions are derived from the gcov data itself, so a newly added
function shows up without touching this script. Functions never executed are listed separately. A SIMD variant the
current CPU cannot run (e.g. the AVX-512 kernel on an AVX2 machine) is reported as such, not mistaken for a coverage
gap that the matrix failed to close.

Exit status is non-zero when
  - the differential test saw any mismatch,
  - the matrix ran over its time budget,
  - or a decoder function that *should* have run has an execution count of zero.

Usage: difftest_coverage.py GCOV_JSON_GZ SIDECAR_JSON
"""
import gzip
import json
import sys

# Source files that make up the library (coverage of test/difftest.c itself is not part of the report).
LIBRARY_SUFFIXES = ("qoamax.c", "qoamax_lanes.h", "qoamax_beam.h")

# A function counts as part of the decoder if its name matches one of these. The patterns follow the library's
# naming conventions rather than a hard-coded function table, so a newly added decode function is recognized on its
# own. The "_lanes" suffix alone does NOT qualify: the SIMD scalefactor/ULTRAFAST *encoder* searches live in
# qoamax_lanes.h too (qoamax_hillclimb_lanes*, qoamax_encode_ultrafast_lanes*), so only the decode kernels match.
DECODER_PREFIXES = (
    "qoamax_decode",        # qoamax_decode_slice, qoamax_decode_streams, qoamax_decode_lanes*
    "qoamax__decode",       # qoamax__decode_streams, qoamax__decode_stream_scalar
    "qoamax_decoder_",      # the file-format decoder (init/read/frame header/...)
)


def is_library_file(path):
    return path.endswith(LIBRARY_SUFFIXES)


def is_decoder_function(name):
    return name.startswith(DECODER_PREFIXES)


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} GCOV_JSON_GZ SIDECAR_JSON", file=sys.stderr)
        return 2
    gcov_path, sidecar_path = argv[1], argv[2]

    with gzip.open(gcov_path, "rt") as fh:
        gcov = json.load(fh)
    with open(sidecar_path, "rt") as fh:
        sidecar = json.load(fh)

    uncallable = set(sidecar.get("uncallable_variants", []))

    # Collect every library function with its execution count and basic-block coverage.
    functions = []  # (name, source, exec_count, blocks_executed, blocks_total)
    for entry in gcov["files"]:
        if not is_library_file(entry["file"]):
            continue
        source = entry["file"].split("/")[-1]
        for fn in entry["functions"]:
            functions.append(
                (
                    fn["name"],
                    source,
                    int(fn["execution_count"]),
                    int(fn["blocks_executed"]),
                    int(fn["blocks"]),
                )
            )

    decoder = [f for f in functions if is_decoder_function(f[0])]
    decoder.sort(key=lambda f: (f[1], f[0]))

    covered = [f for f in decoder if f[2] > 0]
    zero = [f for f in decoder if f[2] == 0]

    # Split the never-executed functions into ones the CPU physically could not run and real coverage gaps. The
    # sidecar names the decoder variants; any other *__x16 function is the same AVX-512 feature gate (the encoder
    # searches), recognized by suffix so it needs no registration.
    def cpu_blocked(name):
        return name in uncallable or name.endswith("__x16")

    blocked, gaps = [], []
    for name, source, count, bexec, btot in zero:
        if cpu_blocked(name):
            blocked.append((name, source))
        else:
            gaps.append((name, source, bexec, btot))

    print("=" * 78)
    print("Differential test: SIMD vs. scalar decoder")
    print("=" * 78)
    print(f"samples compared       : {sidecar['samples_compared']}")
    print(f"differing samples      : {sidecar['mismatches']} (required 0)")
    print(f"max sample difference  : {sidecar['max_sample_diff']} (required 0)")
    print(f"elapsed / budget       : {sidecar['elapsed_seconds']:.2f}s / {sidecar['time_budget_seconds']:.0f}s")
    over_budget = sidecar["elapsed_seconds"] > sidecar["time_budget_seconds"]
    if over_budget:
        print("                       : OVER BUDGET")

    print("\n" + "-" * 78)
    print(f"Decoder function coverage ({len(covered)}/{len(decoder)} functions executed)")
    print("-" * 78)
    print(f"{'function':<46} {'source':<16} {'calls':>10}  {'blocks':>8}")
    for name, source, count, bexec, btot in covered:
        print(f"{name:<46} {source:<16} {count:>10}  {bexec:>3}/{btot:<4}")

    print("\n" + "-" * 78)
    print("Decoder functions never executed")
    print("-" * 78)
    if not zero:
        print("(none — every decoder function ran)")
    else:
        if gaps:
            print("uncovered by the matrix:")
            for name, source, bexec, btot in gaps:
                print(f"  {name:<44} {source}  (blocks {bexec}/{btot})")
        if blocked:
            print("not callable on this CPU (feature absent, expected):")
            for name, source in blocked:
                print(f"  {name:<44} {source}")

    # Also surface non-decoder library functions that never ran, so the report stays a complete per-function view.
    other_zero = [f for f in functions if not is_decoder_function(f[0]) and f[2] == 0]
    if other_zero:
        print("\n" + "-" * 78)
        print("Other library functions never executed (encoder-only / unused on this CPU)")
        print("-" * 78)
        for name, source, count, bexec, btot in sorted(other_zero, key=lambda f: (f[1], f[0])):
            tag = " [CPU feature absent]" if cpu_blocked(name) else ""
            print(f"  {name:<44} {source}{tag}")

    print("=" * 78)
    ok = (
        sidecar["mismatches"] == 0
        and sidecar["max_sample_diff"] == 0
        and not over_budget
        and not gaps
    )
    if ok:
        print("RESULT: pass — SIMD and scalar outputs are identical; all runnable decoder functions are covered.")
    else:
        print("RESULT: FAIL")
        if sidecar["mismatches"] != 0 or sidecar["max_sample_diff"] != 0:
            print("  - SIMD/scalar output mismatch")
        if over_budget:
            print("  - time budget exceeded")
        if gaps:
            print(f"  - {len(gaps)} decoder function(s) not covered by the matrix")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
