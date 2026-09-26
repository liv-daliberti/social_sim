"""Give every process its own Triton cache directory.

Two vLLM tensor-parallel workers compiling the same kernel race on Triton's
write-temp-then-rename step and one loses with ENOTEMPTY ("Directory not
empty"), which kills the worker and leaves the other hanging in the
learner-actor rendezvous until it times out. TRITON_CACHE_DIR is exported once
per job by .runtime/oat_env.sh, so every process inherits the same string; this
runs at interpreter startup, before Triton is imported, and appends the pid.

Only active when the parent set TRITON_CACHE_DIR, so it cannot invent a cache
location of its own, and it is a no-op for single-worker runs.
"""
import os

# Install the debug handler here rather than only in the experiment module.
# Launchpad and vLLM start fresh interpreters which do not reliably import that
# module, but Python imports sitecustomize in every child before application
# code. Keeping the file handle global is required for faulthandler's lifetime.
_stack_handle = None
try:
    import faulthandler
    import signal

    _stack_path = "/tmp/oat_stacks_%s_%d.txt" % (
        os.environ.get("SLURM_JOB_ID", "na"), os.getpid())
    _stack_handle = open(_stack_path, "w")
    faulthandler.register(
        signal.SIGUSR1, file=_stack_handle, all_threads=True, chain=False)
except Exception:
    pass

_base = os.environ.get("TRITON_CACHE_DIR")
if _base:
    _per_process = os.path.join(_base, "p%d" % os.getpid())
    try:
        os.makedirs(_per_process, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = _per_process
    except OSError:
        pass
