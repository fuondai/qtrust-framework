from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch


class ModelAdapter:
    """Adapter to expose model parameters for FL/SMPC aggregation and canary update.

    Currently supports RainbowAgent; other agents may return no-op.
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        # cache shapes/order for flatten/unflatten
        self._param_shapes: Optional[List[Tuple[int, ...]]] = None
        self._param_sizes: Optional[List[int]] = None
        self._param_names: Optional[List[str]] = None
        # select full model params for aggregation to avoid proxying
        self._select_all_params = True

    @staticmethod
    def from_agent(agent) -> "ModelAdapter":
        return ModelAdapter(agent)

    def _collect_params(self) -> List[torch.nn.Parameter]:
        # Only aggregate trainable parameters of the online network when available
        if hasattr(self.agent, "online") and isinstance(self.agent.online, torch.nn.Module):
            return [p for p in self.agent.online.parameters() if p.requires_grad]
        return []

    def _ensure_shapes(self) -> None:
        if self._param_shapes is not None:
            return
        params = self._collect_params()
        self._param_shapes = [tuple(p.data.shape) for p in params]
        self._param_sizes = [int(p.numel()) for p in params]
        self._param_names = [n for n, _ in self.agent.online.named_parameters()] if hasattr(self.agent, "online") else []

    def flatten_params(self) -> np.ndarray:
        self._ensure_shapes()
        vecs: List[np.ndarray] = []
        for p in self._collect_params():
            v = p.detach().cpu().view(-1).numpy()
            vecs.append(v)
        if not vecs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(vecs).astype(np.float32)

    def unflatten_into_model(self, flat: np.ndarray) -> None:
        self._ensure_shapes()
        if self._param_shapes is None or self._param_sizes is None:
            return
        offset = 0
        device = next(self.agent.online.parameters()).device if hasattr(self.agent, "online") else torch.device("cpu")
        with torch.no_grad():
            for p, shape, size in zip(self._collect_params(), self._param_shapes, self._param_sizes):
                seg = flat[offset: offset + size]
                offset += size
                tensor = torch.from_numpy(seg.reshape(shape)).to(device)
                p.copy_(tensor)

    def quantize(self, vec: np.ndarray, scale: float, prime: int) -> np.ndarray:
        # symmetric fixed-point quantization around zero into [-(prime//2), +(prime//2)-1]
        q = np.rint(vec.astype(np.float64) * scale).astype(np.int64)
        half = prime // 2
        q_mod = ((q + half) % prime) - half
        return q_mod.astype(np.int64)

    def dequantize(self, qvec: np.ndarray, scale: float, prime: int) -> np.ndarray:
        # inverse of quantize
        q = qvec.astype(np.int64)
        # map to [-(prime//2), +(prime//2)-1]
        half = prime // 2
        q = ((q + half) % prime) - half
        return (q.astype(np.float64) / float(scale)).astype(np.float32)

    def get_local_vector(self) -> np.ndarray:
        return self.flatten_params()

    def apply_candidate(self, candidate_flat: np.ndarray, validate_states: List[np.ndarray]) -> bool:
        """Canary apply: evaluate small validation loss under candidate; accept if better or equal.

        Validation criterion: average negative log-likelihood over candidate vs current for CVaR-tail
        action distribution on provided states.
        """
        if not hasattr(self.agent, "online"):
            return False
        # Save current params
        current = self.flatten_params()
        # Compute baseline metric
        base = self._eval_tail_cvar(validate_states)
        # Apply candidate
        try:
            self.unflatten_into_model(candidate_flat)
            cand = self._eval_tail_cvar(validate_states)
        finally:
            # Revert if rejected
            if cand < base:
                # improvement: keep candidate
                return True
            else:
                self.unflatten_into_model(current)
                return False

    def _eval_tail_cvar(self, states: List[np.ndarray], alpha: Optional[float] = None) -> float:
        import torch.nn.functional as F
        if not states:
            return 0.0
        if alpha is None and hasattr(self.agent.cfg, "rl"):
            alpha = float(self.agent.cfg.rl.cvar_alpha)
        alpha = 0.1 if alpha is None else float(alpha)
        with torch.no_grad():
            s = torch.from_numpy(np.stack(states)).float().to(next(self.agent.online.parameters()).device)
            probs = self.agent.online(s)  # [B, A, atoms]
            atoms = probs.shape[-1]
            idx = int(max(0, min(atoms - 1, int(alpha * (atoms - 1)))))
            tail = probs[:, :, : idx + 1]
            # Use entropy over tail as risk metric (lower better)
            p = tail.clamp_min(1e-6)
            ent = -(p * p.log()).sum(dim=(1, 2))
            return float(ent.mean().item())


