from typing import List, Optional
from pydantic import BaseModel, Field

# --- Translation Models ---


class TranslationItem(BaseModel):
    """Result for a single translation item."""

    original_text: Optional[str] = Field(
        None, description="The source text that was translated (optional)"
    )
    translation: str = Field(
        ..., description="The translated text in the target language"
    )
    thinking_process: Optional[str] = Field(
        None, description="Optional chain-of-thought or reasoning included by the model"
    )


class BatchTranslationResult(BaseModel):
    """Container for batch translation results."""

    translations: List[TranslationItem] = Field(
        ..., description="List of translation items"
    )


# --- Evaluation Models ---


class JudgementItem(BaseModel):
    """Result for a single evaluation judgement."""

    accuracy: float = Field(..., ge=1, le=5, description="Accuracy score from 1-5")
    fluency: float = Field(..., ge=1, le=5, description="Fluency score from 1-5")
    explanation: str = Field(
        ..., description="Detailed explanation for the scores given"
    )


class BatchJudgementResult(BaseModel):
    """Container for batch judgement results."""

    evaluations: List[JudgementItem] = Field(
        ..., description="List of evaluation items"
    )
