import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# =======================
# KV-Cached Multi-Head Attention
# =======================
class KVCachedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_cache_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, seq_len: int):
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        return x.view(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)

        Q = self._shape(self.q_proj(query), seq_len_q)
        K = self._shape(self.k_proj(key), seq_len_k)
        V = self._shape(self.v_proj(value), seq_len_k)

        # Handle KV cache
        if cache is not None:
            K = torch.cat([cache["key"], K], dim=2)  # Concatenate along seq_len
            V = torch.cat([cache["value"], V], dim=2)

        # Update cache
        new_cache = {"key": K.detach(), "value": V.detach()} if cache is not None else None

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if use_causal_mask:
            mask = torch.triu(torch.ones(seq_len_q, K.size(2), device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights if self.training else attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        out = self.out_proj(out)

        return out, new_cache


# =======================
# Attention Debugger
# =======================
class AttentionBugDetector:
    def __init__(self):
        self.detected_bugs = []

    def check_scaling_factor(self, module_class) -> Optional[Dict]:
        src = inspect.getsource(module_class.forward)
        if "/ math.sqrt" not in src:
            return {
                "type": "scaling_factor",
                "severity": "critical",
                "description": "Attention scores not divided by sqrt(head_dim)",
                "suggestion": "Divide scores by math.sqrt(head_dim)"
            }
        return None

    def check_softmax_dimension(self, module_class) -> Optional[Dict]:
        src = inspect.getsource(module_class.forward)
        if "softmax" in src and "dim=-1" not in src:
            return {
                "type": "softmax_dimension",
                "severity": "high",
                "description": "Softmax may not be applied on the last dimension",
                "suggestion": "Use dim=-1 in F.softmax"
            }
        return None

    def run_analysis(self, module_class):
        print("="*60)
        print("Attention Debugger Report")
        print("="*60)

        bugs = []
        for check in [self.check_scaling_factor, self.check_softmax_dimension]:
            result = check(module_class)
            if result:
                bugs.append(result)

        if not bugs:
            print("✓ No obvious bugs detected in code inspection.")
        else:
            for bug in bugs:
                print(f"✗ [{bug['severity'].upper()}] {bug['type']}")
                print(f"  Issue: {bug['description']}")
                print(f"  Suggestion: {bug['suggestion']}\n")
        print("="*60)


# =======================
# Runtime Attention Validator
# =======================
class AttentionValidator:
    def __init__(self, tolerance: float = 1e-6):
        self.tol = tolerance

    def validate_attention_weights(self, attn_weights: torch.Tensor) -> Tuple[bool, str]:
        sums = attn_weights.sum(dim=-1)
        if torch.allclose(sums, torch.ones_like(sums), atol=self.tol):
            return True, "Attention weights sum to 1 along last dimension."
        else:
            return False, "Attention weights do NOT sum to 1 along last dimension."

    def validate_causal_mask(self, attn_weights: torch.Tensor) -> Tuple[bool, str]:
        seq_len = attn_weights.size(-1)
        for i in range(seq_len):
            if torch.any(attn_weights[..., i, i+1:] > self.tol):
                return False, "Causal mask not correctly applied."
        return True, "Causal mask applied correctly."

    def validate_output_shape(self, output: torch.Tensor, query: torch.Tensor) -> Tuple[bool, str]:
        if output.shape == query.shape:
            return True, "Output shape matches query shape."
        return False, f"Output shape {output.shape} does not match query shape {query.shape}."


# =======================
# Example Usage
# =======================
def main():
    torch.manual_seed(42)
    model = KVCachedMultiHeadAttention(d_model=16, num_heads=2, max_cache_len=128, dropout=0.0)

    query = torch.randn(1, 4, 16)
    key = query.clone()
    value = query.clone()

    output, cache = model(query, key, value, cache=None, use_causal_mask=True)
    print("Output shape:", output.shape)

    # Debugger
    detector = AttentionBugDetector()
    detector.run_analysis(KVCachedMultiHeadAttention)

    # Validator
    validator = AttentionValidator()
    attn_weights = F.softmax(torch.randn(1, 2, 4, 4), dim=-1)  # fake weights for demo
    print(validator.validate_attention_weights(attn_weights))
    print(validator.validate_output_shape(output, query))


if __name__ == "__main__":
    main()
