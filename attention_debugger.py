import torch
import torch.nn as nn
import inspect
import ast
from typing import List, Dict, Tuple, Optional
import math


class AttentionBugDetector:
    """
    Static analyzer for attention mechanism code.
    Detects common bugs in KV-Cached Multi-Head Attention.
    """

    def __init__(self):
        self.detected_bugs = []

    def analyze_code(self, module) -> List[Dict[str, any]]:
        """
        Analyze the source code of a module and detect common attention bugs.
        """
        source = inspect.getsource(module)
        tree = ast.parse(source)
        bugs = []

        # Check for scaling factor bug
        if "self.scale" in source and "/self.scale" in source:
            if "math.sqrt" not in source:
                bugs.append({
                    'type': 'scaling_factor',
                    'severity': 'critical',
                    'location': 'forward/_compute_attention_scores',
                    'description': 'Attention scores divided by head_dim instead of sqrt(head_dim)',
                    'suggestion': 'Use math.sqrt(head_dim) instead of head_dim'
                })

        # Check for softmax dimension bug
        if "F.softmax" in source:
            if "dim=2" in source:
                bugs.append({
                    'type': 'softmax_dimension',
                    'severity': 'high',
                    'location': 'forward',
                    'description': 'Softmax applied on wrong dimension',
                    'suggestion': 'Use dim=-1 (last dimension) for softmax'
                })

        # Check cache concatenation dimension
        if "torch.cat" in source and "dim=2" in source:
            bugs.append({
                'type': 'cache_concat',
                'severity': 'critical',
                'location': 'forward',
                'description': 'Cache concatenation along wrong dimension',
                'suggestion': 'Use dim=1 (sequence dimension) for concatenating cache'
            })

        # Check dropout during inference
        if "self.dropout" in source and "if self.training" not in source:
            bugs.append({
                'type': 'dropout_in_inference',
                'severity': 'medium',
                'location': 'forward',
                'description': 'Dropout applied during evaluation/inference',
                'suggestion': 'Apply dropout only if self.training is True'
            })

        return bugs

    def run_analysis(self, module):
        print("=" * 70)
        print("AI Attention Debugger - Static Analysis Report")
        print("=" * 70)

        bugs = self.analyze_code(module)
        if not bugs:
            print("\n✓ No static bugs detected!")
            return

        print(f"\n✗ Found {len(bugs)} potential bug(s):\n")
        for i, bug in enumerate(bugs, 1):
            severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(bug['severity'], '⚪')
            print(f"{i}. {severity_icon} [{bug['severity'].upper()}] {bug['type']}")
            print(f"   Location: {bug.get('location', 'unknown')}")
            print(f"   Issue: {bug['description']}")
            print(f"   Fix: {bug['suggestion']}\n")
        print("=" * 70)


class AttentionValidator:
    """
    Runtime validator for attention computation.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_attention_weights(self, attention_weights: torch.Tensor) -> Tuple[bool, str]:
        sum_along_keys = attention_weights.sum(dim=-1)
        if not torch.allclose(sum_along_keys, torch.ones_like(sum_along_keys), atol=self.tolerance):
            return False, "Attention weights do not sum to 1 along key dimension"
        return True, "Attention weights sum to 1 ✅"

    def validate_output_shape(self, output: torch.Tensor, query: torch.Tensor) -> Tuple[bool, str]:
        if output.shape != query.shape:
            return False, f"Output shape {output.shape} does not match query shape {query.shape}"
        return True, "Output shape matches query ✅"

    def validate_cache_shapes(self, cache: Dict[str, torch.Tensor], expected_seq_len: int) -> Tuple[bool, str]:
        if cache is None:
            return True, "Cache is None, nothing to validate"
        for k in ['key', 'value']:
            if cache.get(k) is not None and cache[k].shape[2] != expected_seq_len:
                return False, f"Cache '{k}' sequence length {cache[k].shape[2]} != expected {expected_seq_len}"
        return True, "Cache shapes are correct ✅"

    def validate_causal_mask(self, attention_weights: torch.Tensor, seq_len: int) -> Tuple[bool, str]:
        # upper triangle without diagonal
        upper = torch.triu(attention_weights, diagonal=1)
        if torch.any(upper > self.tolerance):
            return False, "Causal mask incorrectly applied, future positions have non-zero attention"
        return True, "Causal mask applied correctly ✅"


# -----------------------------
# Example usage with KVCachedMultiHeadAttention
# -----------------------------
if __name__ == "__main__":
    import hard.kv_attention as kv_attention  # Your attention module

    print("\n💻 Running static analysis...\n")
    detector = AttentionBugDetector()
    detector.run_analysis(kv_attention)

    print("\n🧪 Running runtime validation...\n")
    model = kv_attention.KVCachedMultiHeadAttention(d_model=64, num_heads=4)
    model.eval()

    batch_size, seq_len, d_model = 2, 8, 64
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    output, cache = model(q, k, v)

    validator = AttentionValidator()
    valid_weights, msg_weights = validator.validate_attention_weights(model._compute_attention_scores(model._split_heads(q), model._split_heads(k)))
    valid_shape, msg_shape = validator.validate_output_shape(output, q)
    valid_cache, msg_cache = validator.validate_cache_shapes(cache, seq_len)
    valid_mask, msg_mask = validator.validate_causal_mask(torch.softmax(model._compute_attention_scores(model._split_heads(q), model._split_heads(k)), dim=-1), seq_len)

    print(f"✅ Attention Weights: {msg_weights}")
    print(f"✅ Output Shape: {msg_shape}")
    print(f"✅ Cache Shapes: {msg_cache}")
    print(f"✅ Causal Mask: {msg_mask}")
