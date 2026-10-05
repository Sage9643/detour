"""Database-level invariants of the interaction log.

These constraints protect the dataset that future evaluation and learned-ranking work
depends on, so they are enforced in PostgreSQL, not only in application code.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from detour.db.models import (
    Feedback,
    Interest,
    RecommendationRun,
    Repository,
    RunCandidate,
    User,
)
from detour.enums import FeedbackType, QueryFamily, RankerVariant, RunStatus
from tests.factories import make_repo, make_run, make_user

pytestmark = pytest.mark.db


def assert_rejected(session: Session, add: Callable[[], object]) -> None:
    """The write must violate a constraint; the outer test transaction stays usable."""
    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(add())
        session.flush()


def candidate(run: RecommendationRun, repo: Repository, **overrides: object) -> RunCandidate:
    fields: dict[str, object] = {
        "run_id": run.id,
        "repo_id": repo.github_id,
        "query_families": ["core"],
        "shown": False,
        "position": None,
    }
    fields.update(overrides)
    return RunCandidate(**fields)


# --- enums and their CHECK constraints must agree -------------------------------------


@pytest.mark.parametrize("ftype", list(FeedbackType))
def test_every_feedback_type_is_accepted(session: Session, ftype: FeedbackType) -> None:
    user, repo = make_user(session), make_repo(session)
    session.add(Feedback(user_id=user.id, repo_id=repo.github_id, type=ftype.value))
    session.flush()


def test_unknown_feedback_type_is_rejected(session: Session) -> None:
    user, repo = make_user(session), make_repo(session)
    assert_rejected(session, lambda: Feedback(user_id=user.id, repo_id=repo.github_id, type="MEH"))


@pytest.mark.parametrize("variant", list(RankerVariant))
@pytest.mark.parametrize("status", list(RunStatus))
def test_every_variant_and_status_is_accepted(
    session: Session, variant: RankerVariant, status: RunStatus
) -> None:
    make_run(session, make_user(session), variant=variant.value, status=status.value)


def test_unknown_variant_is_rejected(session: Session) -> None:
    user = make_user(session)
    assert_rejected(
        session,
        lambda: RecommendationRun(
            user_id=user.id,
            variant="B9",
            engine_version="t",
            config={},
            config_hash="x",
            profile_snapshot={},
            status="ok",
        ),
    )


def test_all_query_families_accepted_together(session: Session) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    session.add(candidate(run, repo, query_families=[f.value for f in QueryFamily]))
    session.flush()


@pytest.mark.parametrize("families", [[], ["core", "telepathy"]])
def test_invalid_query_families_rejected(session: Session, families: list[str]) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    assert_rejected(session, lambda: candidate(run, repo, query_families=families))


# --- run_candidates invariants ----------------------------------------------------------


def test_shown_requires_position_and_vice_versa(session: Session) -> None:
    run = make_run(session, make_user(session))
    r1, r2 = make_repo(session, 1), make_repo(session, 2)
    assert_rejected(session, lambda: candidate(run, r1, shown=True, position=None))
    assert_rejected(session, lambda: candidate(run, r2, shown=False, position=3))


def test_positions_unique_within_run_but_unshown_rows_unlimited(session: Session) -> None:
    run = make_run(session, make_user(session))
    repos = [make_repo(session, i) for i in range(1, 5)]
    session.add(candidate(run, repos[0], shown=True, position=0))
    # Many unshown candidates (NULL position) in one run are fine: partial unique index.
    session.add_all([candidate(run, r) for r in repos[1:3]])
    session.flush()
    assert_rejected(session, lambda: candidate(run, repos[3], shown=True, position=0))


def test_same_position_allowed_in_different_runs(session: Session) -> None:
    user, repo = make_user(session), make_repo(session)
    for _ in range(2):
        session.add(candidate(make_run(session, user), repo, shown=True, position=0))
    session.flush()


def test_repo_appears_once_per_run(session: Session) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    session.add(candidate(run, repo))
    session.flush()
    assert_rejected(session, lambda: candidate(run, repo))


def test_filtered_candidate_cannot_be_shown(session: Session) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    assert_rejected(
        session,
        lambda: candidate(run, repo, shown=True, position=0, filter_reason="already_known"),
    )


def test_negative_position_rejected(session: Session) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    assert_rejected(session, lambda: candidate(run, repo, shown=True, position=-1))


def test_features_and_scores_round_trip(session: Session) -> None:
    run, repo = make_run(session, make_user(session)), make_repo(session)
    c = candidate(
        run,
        repo,
        shown=True,
        position=0,
        relevance_raw=0.42,
        novelty=0.7,
        features={"stars_at_request": 812, "language_match": True},
        reasons={"matched_interest": "Distributed Systems"},
    )
    session.add(c)
    session.flush()
    session.expire_all()
    loaded = session.get(RunCandidate, c.id)
    assert loaded is not None
    assert loaded.relevance_raw == pytest.approx(0.42)
    assert loaded.features == {"stars_at_request": 812, "language_match": True}


# --- interests ----------------------------------------------------------------------------


def test_interest_labels_unique_per_user_case_insensitive(session: Session) -> None:
    user, other = make_user(session), make_user(session)
    session.add(Interest(user_id=user.id, label="Distributed Systems"))
    session.add(Interest(user_id=other.id, label="distributed systems"))  # other user: ok
    session.flush()
    assert_rejected(session, lambda: Interest(user_id=user.id, label="DISTRIBUTED SYSTEMS"))


@pytest.mark.parametrize("label", ["", "x" * 101])
def test_interest_label_length_enforced(session: Session, label: str) -> None:
    user = make_user(session)
    assert_rejected(session, lambda: Interest(user_id=user.id, label=label))


def test_interest_weight_must_be_positive(session: Session) -> None:
    user = make_user(session)
    assert_rejected(session, lambda: Interest(user_id=user.id, label="ML", weight=0))


def test_interest_defaults(session: Session) -> None:
    i = Interest(user_id=make_user(session).id, label="ML")
    session.add(i)
    session.flush()
    session.refresh(i)
    assert i.weight == 1.0
    assert i.active is True


# --- referential behaviour that protects the log --------------------------------------


def test_deleting_interest_keeps_candidate_log_with_label_snapshot(session: Session) -> None:
    user = make_user(session)
    interest = Interest(user_id=user.id, label="Distributed Systems")
    session.add(interest)
    session.flush()
    run, repo = make_run(session, user), make_repo(session)
    c = candidate(run, repo, matched_interest_id=interest.id, matched_interest_label=interest.label)
    session.add(c)
    session.flush()

    session.delete(interest)
    session.flush()
    session.expire_all()

    loaded = session.get(RunCandidate, c.id)
    assert loaded is not None
    assert loaded.matched_interest_id is None
    assert loaded.matched_interest_label == "Distributed Systems"


def test_deleting_user_cascades_to_their_data_but_keeps_repositories(session: Session) -> None:
    user, repo = make_user(session), make_repo(session)
    session.add(Interest(user_id=user.id, label="ML"))
    run = make_run(session, user)
    session.add(candidate(run, repo, shown=True, position=0))
    session.add(Feedback(user_id=user.id, repo_id=repo.github_id, run_id=run.id, type="INTERESTED"))
    session.flush()

    session.delete(user)
    session.flush()
    session.expire_all()

    for model in (Interest, RecommendationRun, RunCandidate, Feedback):
        assert session.scalars(select(model)).all() == [], model.__name__
    assert session.get(Repository, repo.github_id) is not None
    assert session.scalars(select(User)).all() == []


def test_repository_referenced_by_log_cannot_be_deleted(session: Session) -> None:
    user, repo = make_user(session), make_repo(session)
    session.add(Feedback(user_id=user.id, repo_id=repo.github_id, type="ALREADY_KNOW"))
    session.flush()
    with pytest.raises(IntegrityError), session.begin_nested():
        session.delete(repo)
        session.flush()


def test_feedback_is_append_only_history(session: Session) -> None:
    """A user may change their mind; both events are kept."""
    user, repo = make_user(session), make_repo(session)
    session.add(Feedback(user_id=user.id, repo_id=repo.github_id, type="NOT_INTERESTED"))
    session.add(Feedback(user_id=user.id, repo_id=repo.github_id, type="INTERESTED"))
    session.flush()
    rows = session.scalars(select(Feedback).where(Feedback.user_id == user.id)).all()
    assert sorted(r.type for r in rows) == ["INTERESTED", "NOT_INTERESTED"]


def test_repository_counts_cannot_be_negative(session: Session) -> None:
    assert_rejected(
        session,
        lambda: Repository(
            github_id=99,
            full_name="a/b",
            owner_login="a",
            name="b",
            html_url="https://github.com/a/b",
            stars=-1,
            metadata_fetched_at=datetime.now(UTC),
        ),
    )
