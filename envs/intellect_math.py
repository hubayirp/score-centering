"""intellect-math (Environments Hub) with an enforceable scoring timeout.

Some RL completions drive math_verify's regexes into catastrophic
backtracking, a GIL-holding C loop that its in-process alarms cannot
interrupt; one such call wedges the scoring pool and every later reward is 0.
So each verify runs in a forked child that is killed after KILL_AFTER_SECONDS.
"""
import asyncio
import multiprocessing
import sys
from functools import partial

import verifiers as vf
import intellect_math
from verifiers.rubrics.math_rubric import MathRubric, verify_response
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.thread_utils import unregister_executor

KILL_AFTER_SECONDS = 30.0  # in-child ops keep their own 5s alarms; this catches the rest


def _child_verify(conn, response, answer, max_verify_chars, timeout_seconds):
    conn.send(verify_response(response, answer, max_verify_chars, timeout_seconds))
    conn.close()


def bounded_verify_response(response, answer, max_verify_chars, timeout_seconds):
    """verify_response in a child process that is killed if it hangs."""
    ctx = multiprocessing.get_context("fork")  # worker is single-threaded
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child_verify, args=(child_conn, response, answer, max_verify_chars, timeout_seconds))
    child.start()
    child_conn.close()
    try:
        if parent_conn.poll(KILL_AFTER_SECONDS):
            return parent_conn.recv()
        print(f"math-verify hung; killed. response={response[:500]!r}", file=sys.stderr)
        return 0.0, KILL_AFTER_SECONDS
    except EOFError:  # child died without replying
        return 0.0, 0.0
    finally:
        child.kill()  # no-op if already exited
        child.join(timeout=5)
        parent_conn.close()


class BoundedMathRubric(MathRubric):
    async def correct_answer(self, parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        if len(response) > self.max_verify_chars:
            return 0.0
        verify = partial(bounded_verify_response, response, answer, self.max_verify_chars, int(self.timeout_seconds))
        try:
            reward, elapsed = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(self.executor, verify), timeout=self.HARD_TIMEOUT_SECONDS)
        except Exception as e:  # includes the (backstop) hard timeout
            self.logger.warning(f"math-verify failed: {e!r}")
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
    """strict=True scores completions without a \\boxed{} answer 0 instead of
    parsing the whole completion text (upstream's fallback)."""
    env = intellect_math.load_environment(**kwargs)
    parser = vf.Parser(partial(extract_boxed_answer, strict=True)) if strict else None
    env.rubric = _replace_fragile(env.rubric, parser)
    return env
