"""User, interest, recommendation and feedback endpoints.

The API is shaped around the product loop:
  create user -> set interests -> get recommendations -> give feedback -> get more
"""

import uuid

from fastapi import APIRouter, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from detour.api.deps import EngineDeps, SessionDep, api_error
from detour.api.schemas import (
    CatalogEntry,
    CreateUserRequest,
    FeedbackRequest,
    FeedbackResponse,
    InterestOut,
    PutInterestsRequest,
    RecommendRequest,
    RecommendResponse,
    RunOut,
    UserOut,
)
from detour.db.models import Interest, User
from detour.profile.builder import build_profile
from detour.profile.feedback import FeedbackRejected, record_feedback
from detour.recommend.service import (
    NoInterests,
    RetrievalFailed,
    RunView,
    UserNotFound,
    recommend,
)
from detour.representation.interests import CURATED_LABELS, resolve_interest

router = APIRouter(tags=["users"])


def _user_or_404(session: Session, user_id: uuid.UUID) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise api_error(status.HTTP_404_NOT_FOUND, "user_not_found", "user not found")
    return user


def _interests_out(session: Session, user_id: uuid.UUID) -> list[InterestOut]:
    rows = session.scalars(
        select(Interest)
        .where(Interest.user_id == user_id, Interest.active.is_(True))
        .order_by(Interest.created_at, Interest.label)
    ).all()
    out = []
    for row in rows:
        spec = resolve_interest(row.label, row.expanded_text)
        out.append(
            InterestOut(
                id=row.id,
                label=row.label,
                curated=spec.curated,
                expansion=spec.expansion,
                topics=list(spec.topics),
            )
        )
    return out


@router.get("/interests/catalog", response_model=list[CatalogEntry], tags=["interests"])
def interest_catalog() -> list[CatalogEntry]:
    """Curated interests with hand-written expansions (any other label is also accepted)."""
    return [
        CatalogEntry(label=label, topics=list(resolve_interest(label).topics))
        for label in CURATED_LABELS
    ]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, session: SessionDep) -> UserOut:
    user = User(github_username=body.github_username)
    session.add(user)
    session.flush()
    session.refresh(user)
    return UserOut(
        id=user.id, github_username=user.github_username, created_at=user.created_at, interests=[]
    )


@router.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: uuid.UUID, session: SessionDep, deps: EngineDeps) -> UserOut:
    user = _user_or_404(session, user_id)
    _, snapshot = build_profile(session, user_id, deps.embedder, deps.ranker_cfg)
    return UserOut(
        id=user.id,
        github_username=user.github_username,
        created_at=user.created_at,
        interests=_interests_out(session, user_id),
        profile=snapshot,
    )


@router.put("/users/{user_id}/interests", response_model=list[InterestOut])
def put_interests(
    user_id: uuid.UUID, body: PutInterestsRequest, session: SessionDep
) -> list[InterestOut]:
    """Replace the active interest set.

    Rows are matched case-insensitively and reactivated/deactivated rather than deleted,
    so feedback attributed to an interest keeps its history if it is re-added.
    """
    _user_or_404(session, user_id)
    existing = {
        row.label.lower(): row
        for row in session.scalars(select(Interest).where(Interest.user_id == user_id)).all()
    }
    wanted = {i.label.lower(): i for i in body.interests}
    for key, old in existing.items():
        if key not in wanted:
            old.active = False
    for key, item in wanted.items():
        match = existing.get(key)
        if match is None:
            session.add(Interest(user_id=user_id, label=item.label, expanded_text=item.expansion))
        else:
            match.active = True
            match.label = item.label
            match.expanded_text = item.expansion
            match.updated_at = func.now()
    session.flush()
    return _interests_out(session, user_id)


@router.post(
    "/users/{user_id}/recommendations",
    response_model=RecommendResponse,
    responses={503: {"description": "GitHub unavailable and nothing cached"}},
)
def create_recommendations(
    user_id: uuid.UUID, body: RecommendRequest, session: SessionDep, deps: EngineDeps
) -> RecommendResponse:
    """Run the full pipeline once and log the run (and every scored candidate)."""
    try:
        result = recommend(
            session,
            user_id,
            deps,
            variant=body.variant,
            k=body.k,
            compare_variant=body.compare_variant,
        )
    except UserNotFound as exc:
        raise api_error(404, "user_not_found", "user not found") from exc
    except NoInterests as exc:
        raise api_error(409, "no_interests", "set at least one interest first") from exc
    except RetrievalFailed as exc:
        session.commit()  # keep the failed-run record
        raise api_error(
            503,
            "github_unavailable",
            "GitHub is unavailable or rate-limited and no cached results exist for these "
            f"interests. Try again later. ({exc})",
        ) from exc
    return RecommendResponse(
        run=_run_out(result.primary),
        comparison=_run_out(result.comparison) if result.comparison else None,
    )


@router.post(
    "/users/{user_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_feedback(
    user_id: uuid.UUID, body: FeedbackRequest, session: SessionDep, deps: EngineDeps
) -> FeedbackResponse:
    """Append a feedback event and return the profile it produces."""
    try:
        event = record_feedback(session, user_id, body.repo_id, body.type, body.run_id)
    except FeedbackRejected as exc:
        code = 404 if exc.code.endswith("not_found") else 422
        raise api_error(code, exc.code, str(exc)) from exc
    session.refresh(event)
    _, snapshot = build_profile(session, user_id, deps.embedder, deps.ranker_cfg)
    return FeedbackResponse(
        id=event.id,
        repo_id=event.repo_id,
        type=body.type,
        created_at=event.created_at,
        profile=snapshot,
    )


def _run_out(view: RunView) -> RunOut:
    return RunOut(
        run_id=view.run_id,
        variant=view.variant,
        status=view.status.value,
        created_at=view.created_at,
        items=view.items,
        stats=view.stats,
        profile=view.profile,
        warnings=view.warnings,
    )
