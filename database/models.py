from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()


class CommentPrediction(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True, index=True)
    comment_text = Column(String, nullable=False)
    toxicity_score = Column(Float, nullable=False)
    categories = Column(JSON, nullable=False)
    moderation_status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
