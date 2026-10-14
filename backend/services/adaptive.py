"""Select adaptive KG retrieval depth from hop-classifier output."""

from __future__ import annotations

DEFAULT_CONFIDENCE_THRESHOLD = 0.60
MAX_HOPS = 3


def select_retrieval_depth(
    predicted_hop: int,
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Map a hop prediction and confidence to a retrieval depth in {1, 2, 3}."""
    if predicted_hop not in (1, 2, 3):
        raise ValueError(f"predicted_hop must be 1, 2, or 3, got {predicted_hop}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    if confidence >= threshold:
        depth = predicted_hop
        reason = (
            f"high confidence {confidence:.4f} >= {threshold:.2f}; "
            f"retrieve at predicted {predicted_hop}-hop"
        )
        used_fallback = False
    else:
        depth = min(predicted_hop + 1, MAX_HOPS)
        used_fallback = depth != predicted_hop
        if predicted_hop == MAX_HOPS:
            reason = (
                f"low confidence {confidence:.4f} < {threshold:.2f}; "
                f"predicted 3-hop remains at {MAX_HOPS} hops"
            )
        else:
            reason = (
                f"low confidence {confidence:.4f} < {threshold:.2f}; "
                f"fallback from {predicted_hop}-hop to {depth}-hop retrieval"
            )

    return {
        "selected_retrieval_depth": depth,
        "adaptive_decision_reason": reason,
        "used_fallback": used_fallback,
    }
