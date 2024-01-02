"""
Profilers to collect computational metrics
"""

import cProfile
import contextlib
import io
import pstats
import time
from typing import Mapping, Protocol, runtime_checkable

from armory.logs import log


@runtime_checkable
class Profiler(Protocol):
    @contextlib.contextmanager
    def measure(self, name: str):
        yield

    def results(self) -> Mapping[str, float]:
        ...


class NullProfiler:
    """Profiler that does nothing (no-op)"""

    def __init__(self):
        self.measurement_dict = {}

    @contextlib.contextmanager
    def measure(self, name):
        yield

    def results(self):
        return {}


class BasicProfiler(NullProfiler):
    """Profiler using `time.perf_counter`"""

    @contextlib.contextmanager
    def measure(self, name):
        startTime = time.perf_counter()
        yield
        elapsedTime = time.perf_counter() - startTime

        if name not in self.measurement_dict:
            self.measurement_dict[name] = {
                "execution_count": 0,
                "total_time": 0,
            }
        comp = self.measurement_dict[name]
        comp["execution_count"] += 1
        comp["total_time"] += elapsedTime

    def results(self):
        results = {}
        for name, entry in self.measurement_dict.items():
            if "execution_count" not in entry or "total_time" not in entry:
                log.warning(
                    "Computation resource dictionary entry {name} corrupted, missing data."
                )
                continue
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
        return results


class DeterministicProfiler(NullProfiler):
    """Profiler using cProfile for deterministic profiling"""

    def __init__(self):
        super().__init__()
        log.warning(
            "Using Deterministic profiler. This may reduce timing accuracy and result in a large results file."
        )

    @contextlib.contextmanager
    def measure(self, name):
        pr = cProfile.Profile()
        pr.enable()
        startTime = time.perf_counter()
        yield
        elapsedTime = time.perf_counter() - startTime
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()

        if name not in self.measurement_dict:
            self.measurement_dict[name] = {
                "execution_count": 0,
                "total_time": 0,
                "stats": "",
            }
        comp = self.measurement_dict[name]
        comp["execution_count"] += 1
        comp["total_time"] += elapsedTime
        comp["stats"] += stats

    def results(self):
        results = {}
        for name, entry in self.measurement_dict.items():
            if any(x not in entry for x in ("execution_count", "total_time", "stats")):
                log.warning(
                    "Computation resource dictionary entry {name} corrupted, missing data."
                )
                continue
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
            results[f"{name} profiler stats"] = entry["stats"]
        return results
