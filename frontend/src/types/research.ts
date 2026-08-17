export type ResearchSource = {
  title: string;
  url?: string | null;
  source: string;
  year?: number | null;
  abstract?: string | null;
};

export type ResearchPlan = {
  objective: string;
  steps: string[];
  search_queries: string[];
  preferred_sources: string[];
};

export type ClaimCheck = {
  claim: string;
  supported: boolean;
  evidence: string[];
  rationale: string;
};

/**
 * How the critic score was produced.
 * - `llm`: an LLM checked each claim against the retrieved sources.
 * - `heuristic`: keyword overlap only — no model verified anything.
 * - `empty`: no claims were extracted, so the score carries no information.
 */
export type CriticMethod = "llm" | "heuristic" | "empty";

export type ResearchResponse = {
  run_id: string;
  status: string;
  plan?: ResearchPlan | null;
  summary?: string | null;
  critic_score?: number | null;
  critic_method?: CriticMethod | null;
  claim_checks?: ClaimCheck[];
  report_markdown?: string | null;
  sources: ResearchSource[];
  error?: string | null;
};

export type HistoryItem = {
  id: string;
  query: string;
  status: string;
  created_at: string;
  summary?: string | null;
  critic_score?: number | null;
  critic_method?: CriticMethod | null;
  claim_checks?: ClaimCheck[];
  plan?: ResearchPlan | null;
  sources?: ResearchSource[];
  error?: string | null;
  report?: {
    markdown: string;
    json_summary: Record<string, unknown>;
    pdf_path?: string | null;
  } | null;
};

/**
 * Response from `GET /health`.
 *
 * Only `status` is guaranteed. Everything else is optional because the frontend and
 * backend deploy independently — a freshly deployed UI routinely talks to a backend
 * that predates these fields. Treating a missing field as `false` would make the UI
 * state things about the backend that it has not been told, so absent and false are
 * kept distinct throughout.
 */
export type HealthStatus = {
  status: string;
  llm_provider?: string;
  /** False means every summary comes from the local deterministic fallback. Undefined means the backend did not say. */
  llm_configured?: boolean;
  llm_model?: string;
  embedding_backend?: string;
  vector_store_documents?: number;
  /**
   * Most recent provider failure. Non-null while `llm_configured` is true means a key
   * is present but calls are failing (invalid key, quota, rate limit) — so summaries
   * are silently coming from the local fallback.
   */
  last_llm_error?: string | null;
};
