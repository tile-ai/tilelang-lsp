"""Build the target and the defensive prologue pass list (through LayoutInference).

Stage lists mirror ``tilelang/cuda/pipeline.py``, ``tilelang/cpu/pipeline.py``,
``tilelang/metal/pipeline.py`` and ``tilelang/webgpu/pipeline.py`` — truncated
right after LayoutInference, which is where ``layout_map`` annotations land.

Every pass is constructed inside try/except: installed builds can lag the
source tree (e.g. a missing ``LowerBlackwell2SM`` FFI), and a pass that cannot
be *constructed* is skipped with a warning instead of aborting the analysis.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


def _determine_target(target_str: str):
    try:
        from tilelang.backend.target import determine_target
    except ImportError:  # installed 0.1.x exposes it here
        from tilelang.utils.target import determine_target
    return determine_target(target_str)


def build_target(target_str: str):
    """Resolve a target string ('cuda', 'c', 'llvm', 'metal', ...) to a Target
    with a canonical host, the way engine/lower.py does."""
    from tilelang import tvm
    from tilelang.engine.lower import canon_target_host

    target = _determine_target(target_str)
    target_host = tvm.target.Target(canon_target_host(target, None))
    return tvm.target.Target(target, target_host)


def _cond_helpers():
    helpers = {"force_let_inline": lambda: False,
               "enable_race_check": lambda: False,
               "allow_warp_specialized": lambda target=None: False}
    try:
        from tilelang.backend.pass_pipeline.pipeline_utils import (
            should_enable_race_check,
            should_force_let_inline,
        )
        helpers["force_let_inline"] = should_force_let_inline
        helpers["enable_race_check"] = should_enable_race_check
    except Exception:
        pass
    try:
        from tilelang.cuda.pipeline import allow_warp_specialized
        helpers["allow_warp_specialized"] = allow_warp_specialized
    except Exception:
        pass
    return helpers


def prologue_stages(target) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """(stages, warnings): ordered (name, pass) pairs up to LayoutInference."""
    import tilelang
    from tilelang import tvm
    from tvm import tirx

    warnings: List[str] = []
    helpers = _cond_helpers()
    stages: List[Tuple[str, Any]] = []

    def add(name: str, ctor: Callable[[], Any]):
        try:
            stages.append((name, ctor()))
        except Exception as exc:
            warnings.append(f"pass {name} unavailable: {exc}")

    T = tilelang.transform
    kind = str(getattr(getattr(target, "kind", None), "name", "") or "")

    add("BindTarget", lambda: tirx.transform.BindTarget(target))
    add("MaterializeKernelLaunch", T.MaterializeKernelLaunch)
    try:
        if helpers["force_let_inline"]():
            add("LetInline", T.LetInline)
    except Exception:
        pass
    add("AddWrapperForSingleBufStore", T.AddWrapperForSingleBufStore)
    add("LegalizeNegativeIndex", T.LegalizeNegativeIndex)
    try:
        if helpers["enable_race_check"]():
            add("VerifyParallelLoop", T.VerifyParallelLoop)
    except Exception:
        pass
    add("InjectAssumes", T.InjectAssumes)
    add("Simplify", T.Simplify)
    add("LayoutReducer", T.LayoutReducer)

    if kind == "cuda":
        try:
            if helpers["allow_warp_specialized"](target=target):
                add("ProducerConsumerWarpSpecialized",
                    tilelang.cuda.transform.ProducerConsumerWarpSpecialized)
        except Exception:
            pass
        add("LowerBlackwell2SM", tilelang.cuda.transform.LowerBlackwell2SM)
        add("IfStmtBinding", T.IfStmtBinding)
        add("PipelinePlanning", T.PipelinePlanning)
        add("InjectSoftwarePipeline", T.InjectSoftwarePipeline)
        add("Simplify#2", T.Simplify)
    elif kind in ("metal", "webgpu"):
        add("IfStmtBinding", T.IfStmtBinding)
        add("PipelinePlanning", T.PipelinePlanning)
        add("InjectSoftwarePipeline", T.InjectSoftwarePipeline)
        add("Simplify#2", T.Simplify)
        if kind == "metal":
            try:
                from tilelang.metal.pipeline import MetalFragmentToSimdgroup

                # Module-level function, not a pass object: wrap uniformly.
                add("MetalFragmentToSimdgroup", lambda: (lambda mod: MetalFragmentToSimdgroup(mod)))
            except Exception as exc:
                warnings.append(f"pass MetalFragmentToSimdgroup unavailable: {exc}")
    else:  # c / llvm / others: the CPU prologue shape
        add("IfStmtBinding", T.IfStmtBinding)
        add("Simplify#2", T.Simplify)

    add("LayoutInference", T.LayoutInference)
    return stages, warnings
