import { describe, expect, it } from "vitest";

import { toApiError } from "./api";
import { filterSummary, formatStars, percent, primaryFamily, timeAgo } from "./format";

describe("format helpers", () => {
  it("formats stars compactly", () => {
    expect(formatStars(950)).toBe("950");
    expect(formatStars(1200)).toBe("1.2k");
    expect(formatStars(2000)).toBe("2k");
    expect(formatStars(52326)).toBe("52k");
  });

  it("formats percentages and missing values", () => {
    expect(percent(0.734)).toBe("73%");
    expect(percent(null)).toBe("—");
  });

  it("picks the most exploratory query family", () => {
    expect(primaryFamily(["core"])).toBe("core");
    expect(primaryFamily(["core", "bridge"])).toBe("bridge");
    expect(primaryFamily(["core", "bridge", "adjacent"])).toBe("adjacent");
  });

  it("summarises filter reasons in plain words, largest first", () => {
    expect(filterSummary({ fork: 1, below_relevance_gate: 12 })).toEqual([
      "12 not relevant enough",
      "1 forks",
    ]);
  });

  it("describes data age", () => {
    const now = new Date("2026-10-05T12:00:00Z");
    expect(timeAgo("2026-10-05T11:59:30Z", now)).toBe("just now");
    expect(timeAgo("2026-10-05T11:00:00Z", now)).toBe("1 h ago");
    expect(timeAgo(null, now)).toBe("unknown");
  });
});

describe("API errors", () => {
  it("reads Detour error bodies", () => {
    const e = toApiError(503, { detail: { code: "github_unavailable", message: "GitHub down" } });
    expect(e.status).toBe(503);
    expect(e.code).toBe("github_unavailable");
    expect(e.message).toBe("GitHub down");
  });

  it("reads FastAPI validation errors", () => {
    const e = toApiError(422, { detail: [{ msg: "duplicate interest: ML" }] });
    expect(e.code).toBe("validation");
    expect(e.message).toBe("duplicate interest: ML");
  });
});
