import type { FeedbackType, Recommendation } from "./api";
import { FAMILY_LABEL, formatStars, percent, primaryFamily } from "./format";

interface Props {
  rec: Recommendation;
  feedback?: FeedbackType;
  pending?: boolean;
  compact?: boolean;
  highlight?: boolean;
  onFeedback?: (type: FeedbackType) => void;
}

const ACTIONS: { type: FeedbackType; label: string; done: string }[] = [
  { type: "INTERESTED", label: "Interested", done: "Marked interested" },
  { type: "NOT_INTERESTED", label: "Not for me", done: "Marked not for me" },
  { type: "ALREADY_KNOW", label: "Already know it", done: "Marked as known" },
];

function Signal({ label, value, hint, tone }: { label: string; value: number | null; hint: string; tone: string }) {
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="signal" title={hint}>
      <span className="signal-label">{label}</span>
      <span className={`signal-track tone-${tone}`} aria-hidden="true">
        <span className="signal-fill" style={{ width: `${v * 100}%` }} />
      </span>
      <span className="signal-value">{percent(value)}</span>
    </div>
  );
}

export function RecommendationCard({ rec, feedback, pending, compact, highlight, onFeedback }: Props) {
  const { repo, scores } = rec;
  const family = primaryFamily(rec.query_families);
  return (
    <article className={`card family-${family}${highlight ? " card-highlight" : ""}`}>
      <header className="card-head">
        <div className="card-title">
          <a href={repo.html_url} target="_blank" rel="noreferrer" className="repo-name">
            {repo.full_name}
          </a>
          <p className="card-meta">
            <span>{formatStars(repo.stars)} stars</span>
            {repo.language && <span>{repo.language}</span>}
            <span className={`family family-text-${family}`}>{FAMILY_LABEL[family]}</span>
          </p>
        </div>
      </header>

      {repo.description && <p className="card-desc">{repo.description}</p>}

      {!compact && (
        <>
          <ul className="why">
            {rec.explanation.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>

          <div className="signals">
            <Signal
              label="Relevance"
              value={scores.relevance_norm}
              tone="rel"
              hint={`Percentile among this run's candidates for ${rec.matched_interest ?? "your interests"} (cosine ${scores.relevance_raw?.toFixed(2) ?? "—"})`}
            />
            <Signal
              label="Novelty"
              value={scores.novelty}
              tone="nov"
              hint="Mean of unfamiliarity vs. repos you know, share of new topics, and how little-known it is"
            />
            <Signal
              label="Distinctness"
              value={scores.max_sim_to_selected == null ? null : 1 - scores.max_sim_to_selected}
              tone="div"
              hint="1 − similarity to the most similar repository already above it in this list"
            />
          </div>

          {repo.topics.length > 0 && (
            <ul className="topics" aria-label="Topics">
              {repo.topics.slice(0, 6).map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {onFeedback && (
        <div className="feedback" role="group" aria-label={`Feedback on ${repo.full_name}`}>
          {feedback ? (
            <span className="feedback-done">{ACTIONS.find((a) => a.type === feedback)?.done}</span>
          ) : (
            ACTIONS.map((a) => (
              <button
                key={a.type}
                type="button"
                className={`btn-feedback fb-${a.type.toLowerCase()}`}
                disabled={pending}
                onClick={() => onFeedback(a.type)}
              >
                {a.label}
              </button>
            ))
          )}
        </div>
      )}
    </article>
  );
}
