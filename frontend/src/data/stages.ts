/**
 * Placeholder text shown for a pipeline stage before the backend reports on it.
 *
 * These describe what each stage *will* do; once a progress event arrives it
 * replaces the placeholder. This file replaced `data/mock.ts`, which held invented
 * dashboard metrics ("1.2K papers indexed", "hallucination score 0.12") that were
 * never rendered — plausible-looking numbers with nothing behind them.
 */
export const stageDescriptions: Record<string, string> = {
  planning: "Split the query into search variants and pipeline steps.",
  retrieval: "Search arXiv and Semantic Scholar, then recall related indexed passages.",
  summarizing: "Condense methods, findings, and open problems.",
  critic: "Check each summary claim against the retrieved sources.",
  report: "Export Markdown, JSON, and PDF artifacts.",
};
