"""Request/response schemas for the public API."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from detour.enums import FeedbackType, RankerVariant

MAX_INTERESTS = 6


class ErrorBody(BaseModel):
    code: str
    message: str


class CreateUserRequest(BaseModel):
    github_username: str | None = Field(default=None, max_length=39, pattern=r"^[A-Za-z0-9-]+$")


class InterestIn(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    expansion: str | None = Field(
        default=None, max_length=300, description="Optional override of the curated expansion"
    )

    @field_validator("label")
    @classmethod
    def strip_label(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("label must not be blank")
        return v


class PutInterestsRequest(BaseModel):
    interests: list[InterestIn] = Field(min_length=1, max_length=MAX_INTERESTS)

    @field_validator("interests")
    @classmethod
    def unique_labels(cls, v: list[InterestIn]) -> list[InterestIn]:
        seen: set[str] = set()
        for i in v:
            key = i.label.lower()
            if key in seen:
                raise ValueError(f"duplicate interest: {i.label}")
            seen.add(key)
        return v


class InterestOut(BaseModel):
    id: uuid.UUID
    label: str
    curated: bool
    expansion: str
    topics: list[str]


class UserOut(BaseModel):
    id: uuid.UUID
    github_username: str | None
    created_at: datetime
    interests: list[InterestOut]
    profile: dict[str, Any] | None = None


class RecommendRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=30)
    variant: RankerVariant = RankerVariant.D3_FULL
    compare_variant: RankerVariant | None = Field(
        default=None, description="Also rank the SAME candidate pool with this variant"
    )


class RunOut(BaseModel):
    run_id: uuid.UUID
    variant: RankerVariant
    status: str
    created_at: datetime
    items: list[dict[str, Any]]
    stats: dict[str, Any]
    profile: dict[str, Any]
    warnings: list[str]


class RecommendResponse(BaseModel):
    run: RunOut
    comparison: RunOut | None = None


class FeedbackRequest(BaseModel):
    repo_id: int = Field(gt=0)
    type: FeedbackType
    run_id: uuid.UUID | None = None


class FeedbackResponse(BaseModel):
    id: int
    repo_id: int
    type: FeedbackType
    created_at: datetime
    profile: dict[str, Any]


class CatalogEntry(BaseModel):
    label: str
    topics: list[str]
