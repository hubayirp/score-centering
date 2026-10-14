"""intellect-math (Environments Hub) with an enforceable scoring timeout.

Root cause of the c10/c11 "scorer wedge": on completions with no \\boxed{},
the env's non-strict extractor falls back to the FULL completion text, and
MathRubric's verify worker runs math_verify's latex regexes over ~8k chars of
arbitrary RL-generated prose. Some strings drive stdlib-re into catastrophic
backtracking — a GIL-holding C loop where math_verify's 5s SIGALRM can never
fire — and the 120s hard timeout only abandons the future, so the stuck call
occupies the singleton pool worker forever and all later verifies score 0.

A timeout over adversarial input is only enforceable as process death, so
each verify runs in a child forked from the pool worker (single-threaded,
sympy already loaded: ~5ms). The in-child 5s alarms still do the soft
bounding — scores are bit-identical wherever the current code terminates —
and on breach the worker SIGKILLs the child, logs the poison input in full
(for the upstream regex fix), and returns 0.0, which is exactly what the
alarm always intended.
"""
import asyncio
import faulthandler
import multiprocessing
import sys
from functools import partial

import verifiers as vf
import intellect_math
from verifiers.rubrics.math_rubric import MathRubric, verify_response
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.thread_utils import unregister_executor

# In-child parse/verify ops keep their own 5s alarms; the kill bound only
# catches alarm-immune C loops. 3 ops x 5s + margin.
KILL_AFTER_SECONDS = 30.0


def _child_verify(conn, response, answer, max_verify_chars, timeout_seconds):
    conn.send(verify_response(response, answer, max_verify_chars, timeout_seconds))
    conn.close()


def bounded_verify_response(response, answer, max_verify_chars, timeout_seconds):
    """Drop-in for verify_response with an enforceable time bound.

    Runs in the pool worker. Dumps all thread stacks to stderr if anything
    here stalls 60s (faulthandler needs no GIL, so it captures C-stuck
    frames — the diagnosis path for any residual hang).
    """
    faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
    ctx = multiprocessing.get_context("fork")  # worker is single-threaded
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    child = ctx.Process(
        target=_child_verify,
        args=(child_conn, response, answer, max_verify_chars, timeout_seconds),
    )
    child.start()
    child_conn.close()
    try:
        if parent_conn.poll(KILL_AFTER_SECONDS):
            return parent_conn.recv()
        print(
            f"math-verify exceeded {KILL_AFTER_SECONDS:.0f}s (alarm-immune "
            f"hang) — killing verify child. POISON response={response!r} "
            f"answer={answer!r}",
            file=sys.stderr, flush=True)
        return 0.0, KILL_AFTER_SECONDS
    except EOFError:  # child died without replying (crash/OOM-kill)
        print(f"math-verify child died. response={response[:500]!r}",
              file=sys.stderr, flush=True)
        return 0.0, 0.0
    finally:
        child.kill()  # no-op if already exited
        child.join(timeout=5)
        parent_conn.close()
        faulthandler.cancel_dump_traceback_later()


class BoundedMathRubric(MathRubric):
    async def correct_answer(self, parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        if len(response) > self.max_verify_chars:
            return 0.0
        try:
            reward, elapsed = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    self.executor, bounded_verify_response, response, answer,
                    self.max_verify_chars, int(self.timeout_seconds)),
                timeout=self.HARD_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:  # backstop; should not fire anymore
            self.logger.warning(
                f"math-verify hard timeout DESPITE kill bound — investigate. "
                f"response={response[:500]!r}")
            return 0.0
        except Exception as e:
            self.logger.warning(f"math-verify failed: {e}")
            return 0.0
        if elapsed > self.timeout_seconds:
            return 0.0
        return reward


def _replace_fragile(rubric, parser=None):
    if isinstance(rubric, MathRubric):
        bounded = BoundedMathRubric(parser=parser or rubric.parser)
        unregister_executor(rubric.executor_name)
        rubric.executor.shutdown(wait=False)  # workers spawn lazily: none yet
        return bounded
    if hasattr(rubric, "rubrics"):  # RubricGroup: swap members in place
        rubric.rubrics = [_replace_fragile(r, parser) for r in rubric.rubrics]
    return rubric


def load_environment(strict=False, **kwargs):
    """strict=True: completions with no \\boxed{} score 0.0 without ever
    entering the latex parser. Upstream (strict=False) falls back to parsing
    the full completion text — the wedge exposure class, and an occasional
    rescue of unboxed-but-correct answers; keep False to reproduce c10/c11
    wave-1 reward semantics."""
    env = intellect_math.load_environment(**kwargs)
    parser = None
    if strict:
        parser = vf.Parser(partial(extract_boxed_answer, strict=True))
        env.parser = parser  # train_rl's frac_parsed reads env.parser
    env.rubric = _replace_fragile(env.rubric, parser)
    replaced = env.rubric if isinstance(env.rubric, BoundedMathRubric) else [
        r for r in getattr(env.rubric, "rubrics", []) if isinstance(r, BoundedMathRubric)]
    assert replaced, f"no MathRubric found inside {type(env.rubric).__name__}"
    return env
