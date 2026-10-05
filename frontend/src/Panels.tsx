import type { ProfileSnapshot, RunOut } from "./api";
import { filterSummary, searchSummary, timeAgo } from "./format";

export function RunPanel({ run }: { run: RunOut }) {
  const s = run.stats;
  const searches = searchSummary(s);
  const filtered = filterSummary(s.filtered);
  return (
    <section className="panel" aria-labelledby="run-heading">
      <h2 id="run-heading">How this list was made</h2>
      <ol className="funnel">
        <li>
          <strong>{s.queries.length}</strong> GitHub searches
          <span className="funnel-note">
            {searches.live} live, {searches.cached} from cache
            {searches.failed > 0 && `, ${searches.failed} failed`}
          </span>
        </li>
        <li>
          <strong>{s.candidates}</strong> candidate repositories
          <span className="funnel-note">
            {s.by_family.core ?? 0} from interest searches, {s.by_family.bridge ?? 0} from bridges,{" "}
            {s.by_family.adjacent ?? 0} from exploration
          </span>
        </li>
        <li>
          <strong>{s.eligible}</strong> passed filters and the relevance check
          {filtered.length > 0 && <span className="funnel-note">Removed: {filtered.join(", ")}</span>}
        </li>
        <li>
          <strong>{s.shown}</strong> picked for novelty and variety
        </li>
      </ol>
      <p className="panel-foot">
        Live from GitHub, data fetched {timeAgo(s.data_fetched_at)}. Embeddings: {s.embedding_model} (
        {s.embedding_cache.hits} reused, {s.embedding_cache.misses} computed). Took{" "}
        {Math.round(s.timings_ms.total_ms ?? 0)} ms.
      </p>
    </section>
  );
}

export function ProfilePanel({ profile }: { profile: ProfileSnapshot }) {
  return (
    <section className="panel" aria-labelledby="profile-heading">
      <h2 id="profile-heading">What Detour knows about you</h2>
      <ul className="weights">
        {profile.interests.map((i) => (
          <li key={i.id}>
            <span className="weight-label">{i.label}</span>
            <span className="weight-track" aria-hidden="true">
              <span className="weight-fill" style={{ width: `${i.effective_weight * 100}%` }} />
            </span>
            <span className="weight-note">
              {i.effective_weight < 1
                ? `weight ${i.effective_weight.toFixed(2)} after ${i.rejections} “not for me”`
                : i.rejections > 0
                  ? `full weight; ${i.rejections} “not for me” so far (lowered from the 3rd)`
                  : "full weight"}
            </span>
          </li>
        ))}
      </ul>
      <dl className="facts">
        <div>
          <dt>Repos you know</dt>
          <dd>{profile.known_count}</dd>
        </div>
        <div>
          <dt>Liked</dt>
          <dd>{profile.positives.length}</dd>
        </div>
        <div>
          <dt>Not for you</dt>
          <dd>{profile.negatives.length}</dd>
        </div>
      </dl>
      <p className="panel-foot">
        {profile.cold_start
          ? "No familiarity data yet: novelty is judged by topics and popularity only. Mark repos you already know to sharpen it."
          : "Novelty now also measures distance from the repositories you've marked as known or liked."}
      </p>
    </section>
  );
}
