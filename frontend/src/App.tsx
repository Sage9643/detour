import { useCallback, useEffect, useState } from "react";

import {
  api,
  ApiError,
  type CatalogEntry,
  type FeedbackType,
  type ProfileSnapshot,
  type RecommendResponse,
} from "./api";
import { InterestPicker } from "./InterestPicker";
import { ProfilePanel, RunPanel } from "./Panels";
import { RecommendationCard } from "./RecommendationCard";

const USER_KEY = "detour.userId";
const K = 8;

function readUserId(): string | null {
  try {
    return localStorage.getItem(USER_KEY);
  } catch {
    return null;
  }
}

function writeUserId(id: string | null) {
  try {
    if (id) localStorage.setItem(USER_KEY, id);
    else localStorage.removeItem(USER_KEY);
  } catch {
    /* storage unavailable: the session still works, it just won't be remembered */
  }
}

function describe(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.code === "github_unavailable")
      return "GitHub is rate-limiting or unreachable and nothing is cached for these interests yet. Wait a minute and try again.";
    return err.message;
  }
  return "Something went wrong. Try again.";
}

export default function App() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [userId, setUserId] = useState<string | null>(readUserId);
  const [selected, setSelected] = useState<string[]>([]);
  const [savedInterests, setSavedInterests] = useState<string[]>([]);
  const [editing, setEditing] = useState(true);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [profile, setProfile] = useState<ProfileSnapshot | null>(null);
  const [feedback, setFeedback] = useState<Record<number, FeedbackType>>({});
  const [pendingRepo, setPendingRepo] = useState<number | null>(null);
  const [compare, setCompare] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.catalog().then(setCatalog).catch((e) => setError(describe(e)));
    const id = readUserId();
    if (!id) return;
    api
      .getUser(id)
      .then((u) => {
        const labels = u.interests.map((i) => i.label);
        setSelected(labels);
        setSavedInterests(labels);
        setProfile(u.profile);
        setEditing(labels.length === 0);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) {
          writeUserId(null);
          setUserId(null);
        } else setError(describe(e));
      });
  }, []);

  const run = useCallback(
    async (labels: string[], withCompare: boolean) => {
      setBusy(true);
      setError(null);
      try {
        let id = userId;
        if (!id) {
          id = (await api.createUser()).id;
          writeUserId(id);
          setUserId(id);
        }
        if (labels.join("\n") !== savedInterests.join("\n")) {
          await api.putInterests(id, labels);
          setSavedInterests(labels);
        }
        const res = await api.recommend(id, K, withCompare);
        setResult(res);
        setProfile(res.run.profile);
        setFeedback({});
        setEditing(false);
        window.scrollTo({ top: 0, behavior: "smooth" });
      } catch (e) {
        setError(describe(e));
      } finally {
        setBusy(false);
      }
    },
    [userId, savedInterests],
  );

  async function sendFeedback(repoId: number, type: FeedbackType) {
    if (!userId || !result) return;
    setPendingRepo(repoId);
    try {
      const res = await api.feedback(userId, repoId, type, result.run.run_id);
      setFeedback((f) => ({ ...f, [repoId]: type }));
      setProfile(res.profile);
    } catch (e) {
      setError(describe(e));
    } finally {
      setPendingRepo(null);
    }
  }

  function startOver() {
    writeUserId(null);
    setUserId(null);
    setSelected([]);
    setSavedInterests([]);
    setResult(null);
    setProfile(null);
    setFeedback({});
    setEditing(true);
  }

  const r = result?.run;
  const cmp = result?.comparison;
  const baselineIds = new Set(cmp?.items.map((i) => i.repo.github_id) ?? []);
  const feedbackCount = Object.keys(feedback).length;

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <svg className="logo" viewBox="0 0 48 48" aria-hidden="true">
            <path d="M6 40 C 6 22, 20 26, 24 16 S 40 8, 42 8" />
            <circle cx="42" cy="8" r="4" />
          </svg>
          <span className="brand-name">Detour</span>
        </div>
        <p className="tagline">
          GitHub repositories that match your interests but sit just off the path you already know.
        </p>
      </header>

      {error && (
        <div className="alert" role="alert">
          {error}
          <button type="button" className="btn-quiet" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {editing ? (
        <main className="setup">
          <h1 className="setup-title">What do you build, study, or tinker with?</h1>
          <p className="setup-lede">
            Detour searches GitHub for your interests, then deliberately ranks up repositories you are
            unlikely to have seen, and keeps the list varied. Every pick says why it was chosen.
          </p>
          <InterestPicker catalog={catalog} selected={selected} onChange={setSelected} disabled={busy} />
          <div className="setup-actions">
            <label className="toggle">
              <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
              Also show a relevance-only list for comparison
            </label>
            <button
              type="button"
              className="btn-primary"
              disabled={busy || selected.length === 0}
              onClick={() => run(selected, compare)}
            >
              {busy ? "Searching GitHub…" : "Find repositories"}
            </button>
          </div>
          {busy && <p className="busy-note">The first search for new interests can take a few seconds.</p>}
        </main>
      ) : (
        <main className="results">
          <div className="results-bar">
            <div>
              <h1 className="results-title">Your detours</h1>
              <p className="results-sub">For {savedInterests.join(", ")}</p>
            </div>
            <div className="results-actions">
              <button type="button" className="btn-quiet" onClick={() => setEditing(true)} disabled={busy}>
                Edit interests
              </button>
              <label className="toggle">
                <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
                Compare with relevance-only
              </label>
              <button
                type="button"
                className="btn-primary"
                onClick={() => run(savedInterests, compare)}
                disabled={busy}
              >
                {busy ? "Searching…" : feedbackCount > 0 ? "Show more, using my feedback" : "Show more"}
              </button>
            </div>
          </div>

          {r && r.status === "degraded" && (
            <div className="notice" role="status">
              Some GitHub searches didn't complete (usually the rate limit), so this list draws on fewer
              candidates than usual{r.warnings.length ? `: ${r.warnings.length} search(es) affected` : ""}.
            </div>
          )}

          {!r && <p className="empty">Choose “Show more” to fetch recommendations.</p>}
          {r && r.items.length === 0 && (
            <p className="empty">
              Nothing new passed the relevance check this time. Try adding a related interest, or come back
              later: repositories you've already seen are skipped for two weeks.
            </p>
          )}

          <div className={cmp ? "columns compare" : "columns"}>
            <div className="feed">
              {cmp && (
                <h2 className="col-head">
                  Detour{" "}
                  <span className="col-note">
                    relevance + novelty + variety; outlined picks are not in the relevance-only list
                  </span>
                </h2>
              )}
              {r?.items.map((rec) => (
                <RecommendationCard
                  key={rec.repo.github_id}
                  rec={rec}
                  feedback={feedback[rec.repo.github_id]}
                  pending={pendingRepo === rec.repo.github_id}
                  highlight={!!cmp && !baselineIds.has(rec.repo.github_id)}
                  onFeedback={(t) => sendFeedback(rec.repo.github_id, t)}
                />
              ))}
            </div>
            {cmp && (
              <div className="feed baseline">
                <h2 className="col-head">
                  Relevance only <span className="col-note">same candidates, ranked by match alone</span>
                </h2>
                {cmp.items.map((rec) => (
                  <RecommendationCard key={rec.repo.github_id} rec={rec} compact />
                ))}
              </div>
            )}
            <aside className="side">
              {profile && <ProfilePanel profile={profile} />}
              {r && <RunPanel run={r} />}
              <button type="button" className="btn-link" onClick={startOver}>
                Start over as a new visitor
              </button>
            </aside>
          </div>
        </main>
      )}
      <footer className="foot">
        Detour ranks with pretrained text embeddings and transparent rules; nothing here is generated by a
        language model.
      </footer>
    </div>
  );
}
