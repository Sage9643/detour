"""Recording feedback events (append-only)."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from detour.db.models import Feedback, RecommendationRun, Repository, RunCandidate, User
from detour.enums import FeedbackType


class FeedbackRejected(Exception):
    """The event is inconsistent with the log (unknown repo, foreign run, not shown...)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def record_feedback(
    session: Session,
    user_id: uuid.UUID,
    repo_id: int,
    ftype: FeedbackType,
    run_id: uuid.UUID | None,
) -> Feedback:
    if session.get(User, user_id) is None:
        raise FeedbackRejected("user_not_found", "user not found")
    if session.get(Repository, repo_id) is None:
        raise FeedbackRejected("repo_not_found", "repository not found")
    if run_id is not None:
        run = session.get(RecommendationRun, run_id)
        if run is None or run.user_id != user_id:
            raise FeedbackRejected("run_not_found", "recommendation run not found for this user")
        shown = session.scalar(
            select(RunCandidate.shown).where(
                RunCandidate.run_id == run_id, RunCandidate.repo_id == repo_id
            )
        )
        if not shown:
            raise FeedbackRejected("not_shown", "repository was not shown in this run")
    event = Feedback(user_id=user_id, repo_id=repo_id, run_id=run_id, type=ftype.value)
    session.add(event)
    session.flush()
    return event
