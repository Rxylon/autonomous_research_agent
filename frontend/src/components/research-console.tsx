"use client";

import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ArrowUpRight, Check, Download, ExternalLink, LoaderCircle, X } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Stepper } from "@/components/ui/stepper";
import { Textarea } from "@/components/ui/textarea";
import { stageDescriptions } from "@/data/stages";
import {
  backendUrl,
  backendWebSocketUrl,
  fetchHealth,
  reportDownloadUrl,
  submitResearchQuery,
} from "@/services/api";
import type { ClaimCheck, CriticMethod, ResearchResponse } from "@/types/research";

const defaultPrompt = "Find recent advances in multimodal deepfake detection and summarize major approaches.";

type ProgressEvent = {
  type: "progress" | "result" | "error";
  stage: string;
  status: string;
  message: string;
  details?: Record<string, unknown>;
  result?: ResearchResponse;
};

const stageOrder = ["planning", "retrieval", "summarizing", "critic", "report"] as const;

/* ── Animation config ─────────────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const sectionStagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1, delayChildren: 0.15 } },
};

const sectionItem = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease } },
};

/**
 * The critic score means different things depending on how it was produced, so it is
 * never shown as a bare number. A 1.00 from the keyword heuristic is not evidence of
 * anything, and presenting it identically to a model-verified 1.00 would mislead.
 */
const criticMethodLabel: Record<CriticMethod, { label: string; tone: "good" | "warn" | "muted" }> = {
  llm: { label: "Model-verified against sources", tone: "good" },
  heuristic: { label: "Keyword heuristic — not model-verified", tone: "warn" },
  empty: { label: "No claims extracted — score carries no information", tone: "muted" },
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function ResearchConsole() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Surfaced as a banner: on a free-tier deployment with no key, every summary comes
  // from the local fallback, and the reader deserves to know that up front.
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
    staleTime: 60_000,
  });

  const isRunning = streaming;

  const stepperSteps = useMemo(() => {
    const eventMap = new Map(events.map((event) => [event.stage, event]));
    return stageOrder.map((stage) => {
      const event = eventMap.get(stage);
      return {
        stage,
        status: (event?.status ?? "pending") as "pending" | "running" | "complete",
        message: event?.message ?? stageDescriptions[stage] ?? "Waiting to start.",
      };
    });
  }, [events]);

  const runResearch = useCallback(async (query: string) => {
    setStreaming(true);
    setResult(null);
    setEvents([]);
    setError(null);

    try {
      await new Promise<void>((resolve, reject) => {
        let settled = false;
        const socket = new WebSocket(backendWebSocketUrl());

        const finish = (fn: () => void) => {
          if (settled) return;
          settled = true;
          try {
            socket.close();
          } catch {
            /* already closing */
          }
          fn();
        };

        socket.addEventListener("open", () => {
          socket.send(JSON.stringify({ query }));
        });

        socket.addEventListener("message", (message) => {
          const payload = JSON.parse(message.data as string) as ProgressEvent;
          if (payload.type === "progress") {
            setEvents((current) => [...current.filter((event) => event.stage !== payload.stage), payload]);
          }
          if (payload.type === "result" && payload.result) {
            setResult(payload.result);
            finish(resolve);
          }
          if (payload.type === "error") {
            finish(() => reject(new Error(payload.message)));
          }
        });

        socket.addEventListener("error", () => {
          finish(() => reject(new Error("WebSocket connection failed")));
        });

        // A closed socket with no result is a failure, not a silent success —
        // otherwise a dropped connection leaves the UI spinning forever.
        socket.addEventListener("close", () => {
          finish(() => reject(new Error("WebSocket closed before a result arrived")));
        });
      });
    } catch (streamError) {
      // A blocked WebSocket (proxies and some hosts drop upgrades) is worth retrying
      // over plain HTTP. If that fails too, the error is real and gets shown.
      try {
        setResult(await submitResearchQuery(query));
      } catch (httpError) {
        setError(
          `${errorMessage(httpError)} (streaming attempt: ${errorMessage(streamError)})`,
        );
      }
    } finally {
      setStreaming(false);
    }
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!isRunning && prompt.trim()) {
        runResearch(prompt);
      }
    }
  }

  const method: CriticMethod = result?.critic_method ?? "empty";
  const methodInfo = criticMethodLabel[method];
  const claimChecks: ClaimCheck[] = result?.claim_checks ?? [];

  return (
    <div className="space-y-32">
      {/* ── Backend capability banner ─────────────────
          Two distinct degraded states, and they need different explanations:
          no key at all, versus a key whose calls are failing. Both mean summaries
          come from the local fallback, but only the second is a bug to chase.

          Note the `=== false`: an older backend omits `llm_configured` entirely, and
          `undefined` means "not reported", not "no key". Warning on absence would
          have this UI assert something it was never told. */}
      {health && (health.llm_configured === false || health.last_llm_error) && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-10 rounded-cards border border-amber-300/60 bg-amber-50 p-16 text-[13px] leading-[1.54] text-amber-900"
        >
          <AlertTriangle className="mt-[2px] h-[15px] w-[15px] shrink-0" />
          {health.llm_configured === false ? (
            <span>
              <strong className="font-[500]">No LLM key configured on this backend.</strong>{" "}
              Retrieval, verification plumbing, and reporting all work, but summaries come from a
              local deterministic fallback and critic scores are keyword heuristics, not model
              verification. Set{" "}
              <code className="font-[family-name:var(--font-geistmono)]">GEMINI_API_KEY</code> or{" "}
              <code className="font-[family-name:var(--font-geistmono)]">OPENAI_API_KEY</code> to
              enable real synthesis.
            </span>
          ) : (
            <span>
              <strong className="font-[500]">
                A key is configured for {health.llm_provider}, but the last call failed.
              </strong>{" "}
              Summaries are falling back to the local deterministic path until this is resolved —
              usually an invalid key, an exhausted quota, or rate limiting.
              <br />
              <code className="mt-4 inline-block break-all font-[family-name:var(--font-geistmono)] text-[11px]">
                {health.last_llm_error}
              </code>
            </span>
          )}
        </motion.div>
      )}

      {/* ── Prompt area ─────────────────────────────── */}
      <div className="space-y-16">
        <motion.div
          whileFocus={{ scale: 1.01 }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
        >
          <Textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe the topic you want to research…"
            size="large"
          />
        </motion.div>

        <div className="flex items-center gap-12">
          <motion.div
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <Button
              variant="primary"
              onClick={() => runResearch(prompt)}
              className="gap-8"
              disabled={isRunning || !prompt.trim()}
            >
              {isRunning ? (
                <LoaderCircle className="h-[16px] w-[16px] animate-spin" />
              ) : (
                <ArrowUpRight className="h-[16px] w-[16px]" />
              )}
              {isRunning ? "Researching…" : "Run research"}
            </Button>
          </motion.div>
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.4 }}
            className="text-[13px] text-silver-mist select-none"
          >
            ⌘↵ to submit
          </motion.span>
        </div>
      </div>

      {/* ── Error state ─────────────────────────────── */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-start gap-10 rounded-cards border border-red-300/60 bg-red-50 p-16 text-[13px] leading-[1.54] text-red-900"
          >
            <X className="mt-[2px] h-[15px] w-[15px] shrink-0" />
            <span>
              <strong className="font-[500]">The research run failed.</strong> {error}
              <br />
              <span className="text-red-800/80">
                Backend: <code className="font-[family-name:var(--font-geistmono)]">{backendUrl}</code>
              </span>
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Horizontal stepper ──────────────────────── */}
      <AnimatePresence>
        {(isRunning || events.length > 0) && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.4, ease }}
            className="rounded-cards bg-surface-card p-20 shadow-xl"
          >
            <Stepper steps={stepperSteps} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Result document ─────────────────────────── */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.article
            key={result.run_id}
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.45, ease }}
            className="rounded-cards bg-surface-elevated p-24 shadow-xl-2 md:p-32 lg:p-40"
          >
            <motion.div variants={sectionStagger} initial="hidden" animate="show">
              {/* § Objective */}
              {result.plan?.objective && (
                <motion.section variants={sectionItem} className="pb-24">
                  <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                    Objective
                  </p>
                  <p className="mt-8 text-[16px] leading-[1.6] text-ink-black">
                    {result.plan.objective}
                  </p>
                  {result.plan.steps.length > 0 && (
                    <ol className="mt-12 space-y-4 text-[14px] leading-[1.54] text-stone-gray list-decimal list-inside">
                      {result.plan.steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ol>
                  )}
                </motion.section>
              )}

              <motion.div variants={sectionItem} className="border-t border-frost-gray/50" />

              {/* § Summary */}
              {result.summary && (
                <motion.section variants={sectionItem} className="py-24">
                  <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                    Summary
                  </p>
                  <p className="mt-8 whitespace-pre-wrap text-[16px] leading-[1.6] text-stone-gray">
                    {result.summary}
                  </p>
                </motion.section>
              )}

              <motion.div variants={sectionItem} className="border-t border-frost-gray/50" />

              {/* § Sources & Critic — side by side on larger screens */}
              <motion.section variants={sectionItem} className="py-24 grid gap-24 md:grid-cols-[1fr_auto]">
                <div>
                  <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                    Sources
                  </p>
                  {result.sources.length > 0 ? (
                    <ul className="mt-12 space-y-8">
                      {result.sources.map((source, index) => (
                        <li key={`${source.title}-${index}`} className="flex items-start gap-8 text-[14px]">
                          <ExternalLink className="mt-[3px] h-[13px] w-[13px] shrink-0 text-silver-mist" />
                          <div>
                            {source.url ? (
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-ink-black font-[450] underline decoration-frost-gray hover:decoration-fire-orange"
                              >
                                {source.title}
                              </a>
                            ) : (
                              <span className="text-ink-black font-[450]">{source.title}</span>
                            )}
                            {source.source && (
                              <span className="ml-8 text-[12px] text-silver-mist">
                                {source.source}
                                {source.year ? `, ${source.year}` : ""}
                              </span>
                            )}
                            {source.abstract && (
                              <p className="mt-4 text-[13px] leading-[1.54] text-slate-gray">{source.abstract}</p>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-8 text-[14px] text-silver-mist">No sources retrieved.</p>
                  )}
                </div>

                {/* Critic score — always shown with how it was derived */}
                <div className="flex flex-col items-start md:items-end md:min-w-[160px]">
                  <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                    Critic score
                  </p>
                  <motion.p
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20, delay: 0.4 }}
                    className="mt-8 text-[40px] font-[500] leading-[1.1] tracking-tight text-ink-black"
                  >
                    {result.critic_score?.toFixed(2) ?? "—"}
                  </motion.p>
                  <p
                    className={`mt-4 max-w-[200px] text-[11px] leading-[1.45] md:text-right ${
                      methodInfo.tone === "good"
                        ? "text-emerald-700"
                        : methodInfo.tone === "warn"
                          ? "text-amber-700"
                          : "text-silver-mist"
                    }`}
                  >
                    {methodInfo.label}
                  </p>
                </div>
              </motion.section>

              {/* § Claim checks */}
              {claimChecks.length > 0 && (
                <>
                  <motion.div variants={sectionItem} className="border-t border-frost-gray/50" />
                  <motion.section variants={sectionItem} className="py-24">
                    <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                      Claim checks
                    </p>
                    <ul className="mt-12 space-y-12">
                      {claimChecks.map((check, index) => (
                        <li key={`${check.claim}-${index}`} className="flex items-start gap-8 text-[14px]">
                          {check.supported ? (
                            <Check className="mt-[3px] h-[13px] w-[13px] shrink-0 text-emerald-600" />
                          ) : (
                            <X className="mt-[3px] h-[13px] w-[13px] shrink-0 text-red-500" />
                          )}
                          <div className="min-w-0">
                            <span className="text-ink-black">{check.claim}</span>
                            {check.rationale && (
                              <p className="mt-4 text-[13px] leading-[1.54] text-slate-gray">{check.rationale}</p>
                            )}
                            {check.evidence.map((evidence, evidenceIndex) => (
                              <p
                                key={evidenceIndex}
                                className="mt-4 border-l-2 border-frost-gray pl-8 text-[12px] leading-[1.5] text-silver-mist"
                              >
                                {evidence}
                              </p>
                            ))}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </motion.section>
                </>
              )}

              {/* § Report + downloads */}
              {result.report_markdown && (
                <>
                  <motion.div variants={sectionItem} className="border-t border-frost-gray/50" />
                  <motion.section variants={sectionItem} className="pt-24">
                    <div className="flex flex-wrap items-center justify-between gap-12">
                      <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                        Report
                      </p>
                      <div className="flex items-center gap-8">
                        {(["md", "json", "pdf"] as const).map((format) => (
                          <a
                            key={format}
                            href={reportDownloadUrl(result.run_id, format)}
                            className="inline-flex items-center gap-4 rounded-menu-items border border-cloud-canvas px-10 py-6 text-[12px] font-[450] text-stone-gray transition-colors hover:border-fire-orange hover:text-fire-orange"
                          >
                            <Download className="h-[12px] w-[12px]" />
                            {format.toUpperCase()}
                          </a>
                        ))}
                      </div>
                    </div>
                    <pre className="mt-12 overflow-auto whitespace-pre-wrap rounded-xl border border-cloud-canvas bg-surface-card p-20 font-[family-name:var(--font-geistmono)] text-[13px] leading-[1.6] text-stone-gray">
                      {result.report_markdown}
                    </pre>
                  </motion.section>
                </>
              )}

              {result.status !== "complete" && (
                <motion.div variants={sectionItem} className="pt-16">
                  <Badge variant="active">{result.status}</Badge>
                </motion.div>
              )}
            </motion.div>
          </motion.article>
        )}
      </AnimatePresence>
    </div>
  );
}
