// Typed client for the Detour API. Shapes mirror backend/src/detour/api/schemas.py.

export type Variant = "B0" | "B1" | "D1" | "D2" | "D3";
export type FeedbackType = "INTERESTED" | "NOT_INTERESTED" | "ALREADY_KNOW";
export type QueryFamily = "core" | "bridge" | "adjacent";

export interface CatalogEntry {
  label: string;
  topics: string[];
}

export interface InterestOut {
  id: string;
  label: string;
  curated: boolean;
  expansion: string;
  topics: string[];
}

export interface ProfileInterest {
  id: string;
  label: string;
  curated: boolean;
  base_weight: number;
  effective_weight: number;
  rejections: number;
  background_similarity: number;
  topics: string[];
}

export interface ProfileSnapshot {
  interests: ProfileInterest[];
  positives: string[];
  negatives: string[];
  known_count: number;
  known_topics_count: number;
  recently_shown_count: number;
  cold_start: boolean;
}

export interface UserOut {
  id: string;
  github_username: string | null;
  created_at: string;
  interests: InterestOut[];
  profile: ProfileSnapshot | null;
}

export interface RepoView {
  github_id: number;
  full_name: string;
  owner: string;
  description: string | null;
  html_url: string;
  language: string | null;
  stars: number;
  forks: number;
  topics: string[];
  pushed_at: string | null;
}

export interface Scores {
  relevance_raw: number | null;
  relevance_margin: number | null;
  relevance_norm: number | null;
  novelty: number | null;
  unfamiliarity: number | null;
  topical_novelty: number | null;
  popularity_novelty: number | null;
  negative_penalty: number | null;
  utility: number | null;
  mmr_score: number | null;
  max_sim_to_selected: number | null;
  pre_rerank_rank: number | null;
  interest_sims: Record<string, number>;
}

export interface Recommendation {
  position: number;
  repo: RepoView;
  scores: Scores;
  matched_interest: string | null;
  query_families: QueryFamily[];
  reasons: Record<string, unknown>;
  explanation: string[];
}

export interface QueryStat {
  family: QueryFamily;
  label: string;
  q: string;
  source: "live" | "cache" | "stale" | "failed";
  result_count: number;
  total_count: number | null;
  latency_ms: number;
  fetched_at: string | null;
  error: string | null;
}

export interface RunStats {
  candidates: number;
  by_family: Partial<Record<QueryFamily, number>>;
  queries: QueryStat[];
  github_calls: number;
  github_authenticated: boolean;
  search_rate_remaining: number | null;
  embedding_cache: { hits: number; misses: number };
  embedding_model: string;
  data_fetched_at: string | null;
  eligible: number;
  filtered: Record<string, number>;
  shown: number;
  timings_ms: Record<string, number>;
}

export interface RunOut {
  run_id: string;
  variant: Variant;
  status: "ok" | "degraded" | "failed";
  created_at: string;
  items: Recommendation[];
  stats: RunStats;
  profile: ProfileSnapshot;
  warnings: string[];
}

export interface RecommendResponse {
  run: RunOut;
  comparison: RunOut | null;
}

export interface FeedbackResponse {
  id: number;
  repo_id: number;
  type: FeedbackType;
  created_at: string;
  profile: ProfileSnapshot;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

const BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", `Can't reach the Detour API at ${BASE}. Is the backend running?`);
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw toApiError(res.status, data);
  }
  return data as T;
}

export function toApiError(status: number, data: unknown): ApiError {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const d = detail as { code?: string; message?: string };
    return new ApiError(status, d.code ?? "error", d.message ?? "Request failed");
  }
  if (Array.isArray(detail)) {
    // FastAPI validation errors
    const msg = detail.map((e: { msg?: string }) => e.msg).filter(Boolean).join("; ");
    return new ApiError(status, "validation", msg || "Invalid input");
  }
  return new ApiError(status, "error", `Request failed (HTTP ${status})`);
}

export const api = {
  catalog: () => request<CatalogEntry[]>("GET", "/interests/catalog"),
  createUser: () => request<UserOut>("POST", "/users", {}),
  getUser: (id: string) => request<UserOut>("GET", `/users/${id}`),
  putInterests: (id: string, labels: string[]) =>
    request<InterestOut[]>("PUT", `/users/${id}/interests`, {
      interests: labels.map((label) => ({ label })),
    }),
  recommend: (id: string, k: number, compare: boolean) =>
    request<RecommendResponse>("POST", `/users/${id}/recommendations`, {
      k,
      variant: "D3",
      compare_variant: compare ? "B1" : null,
    }),
  feedback: (id: string, repoId: number, type: FeedbackType, runId: string) =>
    request<FeedbackResponse>("POST", `/users/${id}/feedback`, {
      repo_id: repoId,
      type,
      run_id: runId,
    }),
};
