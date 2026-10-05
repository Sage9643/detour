import type { QueryFamily, RunStats } from "./api";

export function formatStars(n: number): string {
  if (n >= 1000) {
    const k = n / 1000;
    return `${k >= 10 ? Math.round(k) : k.toFixed(1).replace(/\.0$/, "")}k`;
  }
  return String(n);
}

export function percent(x: number | null | undefined): string {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}

export const FAMILY_LABEL: Record<QueryFamily, string> = {
  core: "Interest search",
  bridge: "Bridges two interests",
  adjacent: "Exploration",
};

/** The most exploratory family that found an item (exploration > bridge > core). */
export function primaryFamily(families: QueryFamily[]): QueryFamily {
  if (families.includes("adjacent")) return "adjacent";
  if (families.includes("bridge")) return "bridge";
  return "core";
}

export const FILTER_LABEL: Record<string, string> = {
  below_relevance_gate: "not relevant enough",
  recently_shown: "shown recently",
  already_known: "already known to you",
  archived: "archived",
  fork: "forks",
  no_text: "no description",
  suspicious_description: "spam-like descriptions",
  below_star_floor: "under 20 stars",
};

export function filterSummary(filtered: Record<string, number>): string[] {
  return Object.entries(filtered)
    .sort((a, b) => b[1] - a[1])
    .map(([reason, n]) => `${n} ${FILTER_LABEL[reason] ?? reason.replace(/_/g, " ")}`);
}

export function searchSummary(stats: RunStats): { live: number; cached: number; failed: number } {
  let live = 0;
  let cached = 0;
  let failed = 0;
  for (const q of stats.queries) {
    if (q.source === "live") live++;
    else if (q.source === "cache" || q.source === "stale") cached++;
    else failed++;
  }
  return { live, cached, failed };
}

export function timeAgo(iso: string | null, now: Date = new Date()): string {
  if (!iso) return "unknown";
  const s = Math.max(0, (now.getTime() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}
