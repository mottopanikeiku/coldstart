"""Saved policies: parameters and the contract needed to replay them in a new process.

A saved policy is plain JSON. Affine policies keep float64 weights; MLP actors keep
float32 tensors, which JSON's float64 numbers hold exactly. Replay runs the protocol's own
evaluator, so batching, seeds and tie-breaking are those of the timed evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time

import numpy as np

from control_clock.protocol import THRESHOLDS, Finished, Policy, RunContext
from control_clock.search import greedy_actions


def affine(weights: np.ndarray, scales: np.ndarray) -> dict:
    return {"kind": "affine-argmax", "weights": weights.tolist(), "scales": scales.tolist()}


def mlp(actor, scales: np.ndarray) -> dict:
    """A Sequential of Linear layers with Tanh between them, as in ppo.ActorCritic."""
    from torch import nn

    linears = [module for module in actor if isinstance(module, nn.Linear)]
    expected = [nn.Linear, nn.Tanh] * (len(linears) - 1) + [nn.Linear]
    if [type(module) for module in actor] != expected:
        raise ValueError("expected Linear layers separated by Tanh")
    layers = [
        {"weight": module.weight.detach().tolist(), "bias": module.bias.detach().tolist()}
        for module in linears
    ]
    return {"kind": "mlp-argmax", "activation": "tanh", "scales": scales.tolist(), "layers": layers}


def build(parameters: dict) -> Policy:
    if parameters["kind"] == "affine-argmax":
        weights = np.asarray(parameters["weights"], dtype=np.float64)
        scales = np.asarray(parameters["scales"], dtype=np.float64)
        return lambda observations: greedy_actions(weights, observations, scales)
    if parameters["kind"] != "mlp-argmax":
        raise ValueError(f"unknown policy kind: {parameters['kind']!r}")
    import torch
    from torch import nn

    modules = []
    for index, layer in enumerate(parameters["layers"]):
        weight = torch.tensor(layer["weight"], dtype=torch.float32)
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32))
        modules.append(linear)
        if index < len(parameters["layers"]) - 1:
            modules.append(nn.Tanh())
    actor = nn.Sequential(*modules)
    scales = np.asarray(parameters["scales"], dtype=np.float32)

    def policy(observations):
        features = torch.from_numpy(np.asarray(observations, dtype=np.float32) / scales)
        with torch.no_grad():
            return actor(features).argmax(dim=-1).numpy()

    return policy


def contract(task: str, kind: str) -> dict:
    """What a replay must match: the task, its pass bar and the packages that compute actions."""
    packages = ["gymnasium", "numpy"] + (["torch"] if kind == "mlp-argmax" else [])
    return {
        "task": task,
        "threshold": THRESHOLDS[task],
        "evaluator": "control_clock.protocol.RunContext.checkpoint",
        "evaluation_policy": "deterministic argmax, lowest-index ties",
        **{name: importlib.metadata.version(name) for name in packages},
    }


def replay(task: str, policy: Policy, seeds, *, compare: Policy | None = None):
    """Evaluate once through the protocol; return returns and a digest of every action.

    With compare, both policies see every observation batch and must choose identical actions.
    """
    digest = hashlib.sha256()

    def recorded(observations):
        actions = np.asarray(policy(observations), dtype=np.int64)
        if compare is not None:
            other = np.asarray(compare(observations), dtype=np.int64)
            if not np.array_equal(actions, other):
                raise RuntimeError("saved policy chose different actions from the live policy")
        digest.update(actions.tobytes())
        return actions

    ctx = RunContext(task, 0, time.perf_counter(), math.inf, eval_seeds=tuple(seeds))
    try:
        ctx.checkpoint(recorded, 0, force=True)
    except Finished:
        pass
    finally:
        ctx.close()
    return ctx.evaluations[-1]["returns"], digest.hexdigest()


def save(ctx: RunContext) -> dict:
    """Export the passing policy and prove the export reproduces the timed evaluation."""
    parameters = ctx.policy_parameters()
    returns, digest = replay(ctx.task, build(parameters), ctx.eval_seeds, compare=ctx.policy)
    if returns != ctx.evaluations[-1]["returns"]:
        raise RuntimeError("saved policy does not reproduce the passing evaluation returns")
    return {
        "parameters": parameters,
        "contract": contract(ctx.task, parameters["kind"]),
        "evaluation_seeds": [ctx.eval_seeds[0], ctx.eval_seeds[-1]],
        "evaluation_actions_sha256": digest,
    }
