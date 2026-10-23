from __future__ import annotations

from sqlalchemy.orm import Session

from database.models import CommentPrediction


def create_prediction_record(
    db: Session,
    *,
    comment_text: str,
    toxicity_score: float,
    categories: dict[str, float],
    moderation_status: str,
) -> CommentPrediction:
    record = CommentPrediction(
        comment_text=comment_text,
        toxicity_score=float(toxicity_score),
        categories=dict(categories),
        moderation_status=moderation_status,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
