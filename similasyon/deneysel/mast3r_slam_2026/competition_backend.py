#!/usr/bin/env python3
"""Memory-bounded headless backend adapted from upstream MASt3R-SLAM main.py.

The graph construction and optimization are unchanged. CUDA allocator caches
are released before large calibrated-GN work so the 6 GB GPU does not lose
hundreds of MiB to cross-process cache fragmentation.
"""

from __future__ import annotations

import time

import torch

from mast3r_slam.config import config, set_global_config
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.frame import Mode
from mast3r_slam.mast3r_utils import load_retriever


def _release_cuda_cache(label: str, index: int | None = None) -> None:
    torch.cuda.empty_cache()
    if index is not None and (index % 10 == 0 or index < 3):
        free, total = torch.cuda.mem_get_info()
        print(
            f"BACKEND_MEMORY label={label} kf={index} "
            f"allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f} "
            f"reserved_mib={torch.cuda.memory_reserved() / 2**20:.1f} "
            f"device_free_mib={free / 2**20:.1f} total_mib={total / 2**20:.1f}",
            flush=True,
        )


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = list(
            retrieval_database.update(
                frame,
                add_after_query=False,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
        )
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx, flush=True)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized", flush=True)
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize", flush=True)

        if successful_loop_closure:
            _release_cuda_cache("pre_reloc_gn", len(keyframes))
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)
    _release_cuda_cache("backend_initialized", 0)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue

        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        kf_idx: list[int] = []
        for j in range(min(1, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if lc_inds:
            print("Database retrieval", idx, ": ", lc_inds, flush=True)

        unique_kf_idx = list(set(kf_idx) - {idx})
        frame_idx = [idx] * len(unique_kf_idx)
        if unique_kf_idx:
            factor_graph.add_factors(
                unique_kf_idx,
                frame_idx,
                config["local_opt"]["min_match_frac"],
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        _release_cuda_cache("pre_global_gn", idx)
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                states.global_optimizer_tasks.pop(0)

