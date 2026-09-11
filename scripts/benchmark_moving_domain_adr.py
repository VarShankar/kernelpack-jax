"""Report cold and warm end-to-end timings for the moving-domain ADR example."""

import argparse
import cProfile
import io
import pstats
from time import perf_counter

import jax

from moving_domain_adr_example import main


def timed_run(h: float, step_count: int) -> tuple[float, dict[str, float | int]]:
    start = perf_counter()
    result = main(h=h, step_count=step_count)
    return perf_counter() - start, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=0.16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    cold_seconds, result = timed_run(args.h, args.steps)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        warm_seconds, _ = timed_run(args.h, args.steps)
        profiler.disable()
    else:
        profiler = None
        warm_seconds, _ = timed_run(args.h, args.steps)
    print(f"backend={jax.default_backend()}")
    print(f"devices={jax.devices()}")
    print(f"h={args.h:.6f}")
    print(f"nodes={result['node_count']}")
    print(f"steps={result['step_count']}")
    print(f"cold_total_seconds={cold_seconds:.6f}")
    print(f"warm_total_seconds={warm_seconds:.6f}")
    print(f"warm_seconds_per_step_upper_bound={warm_seconds / args.steps:.6f}")
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(35)
        print(stream.getvalue())
