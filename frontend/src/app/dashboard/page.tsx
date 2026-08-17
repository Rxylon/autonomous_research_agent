"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { ErrorState } from "@/components/ui/error-state";
import { fetchHealth, fetchHistory } from "@/services/api";

/* ── Animation config ─────────────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1, delayChildren: 0.1 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease } },
};

export default function DashboardPage() {
  const { data, error } = useQuery({ queryKey: ["history"], queryFn: fetchHistory, retry: false });
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: fetchHealth, retry: false });

  const runs = data ?? [];
  const completed = runs.filter((item) => item.status === "complete");

  // Average only over runs that actually have a score. Treating a failed run's
  // missing score as 0 — which this page used to do — silently dragged the average
  // down and made it look like the critic was rejecting work it never saw.
  const scored = runs.filter((item) => typeof item.critic_score === "number");
  const averageCritic =
    scored.length > 0
      ? (scored.reduce((sum, item) => sum + (item.critic_score as number), 0) / scored.length).toFixed(2)
      : "—";

  // Only model-verified scores are meaningful; count them separately so the average
  // above can be read with the right amount of trust.
  const modelVerified = runs.filter((item) => item.critic_method === "llm").length;

  const stats = [
    { label: "Total runs", value: runs.length },
    { label: "Completed", value: completed.length },
    { label: "Avg critic score", value: averageCritic, note: `${scored.length} scored` },
    { label: "Model-verified", value: modelVerified, note: `of ${scored.length} scored` },
  ];

  const recentSummary = runs[0]?.summary ?? "No runs yet.";

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="mx-auto max-w-3xl space-y-32"
    >
      {/* ── Page heading ────────────────────────────── */}
      <motion.div variants={fadeUp}>
        <h1 className="text-[24px] font-[500] leading-[1.33] text-ink-black">Analytics</h1>
        <p className="mt-8 text-[16px] leading-[1.6] text-stone-gray">
          Operational visibility for agent workflows. Every number here is derived from stored run
          history — none of it is illustrative.
        </p>
      </motion.div>

      {error && <ErrorState error={error} what="run analytics" />}

      {/* ── Stats row ───────────────────────────────── */}
      {!error && (
        <motion.div
          variants={fadeUp}
          className="rounded-cards bg-surface-elevated shadow-xl-2 overflow-hidden"
        >
          <div className="grid grid-cols-2 divide-frost-gray/40 sm:grid-cols-4 sm:divide-x">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1, duration: 0.4, ease }}
                className="px-20 py-20 text-center"
              >
                <p className="text-[13px] text-slate-gray">{stat.label}</p>
                <motion.p
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.45 + index * 0.1, type: "spring", stiffness: 300, damping: 20 }}
                  className="mt-4 text-[28px] font-[500] leading-[1.1] tracking-tight text-ink-black"
                >
                  {stat.value}
                </motion.p>
                {stat.note && <p className="mt-2 text-[11px] text-silver-mist">{stat.note}</p>}
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── Backend runtime ─────────────────────────── */}
      {health && (
        <motion.div variants={fadeUp} className="space-y-12">
          <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
            Backend runtime
          </p>
          <dl className="grid gap-x-24 gap-y-8 text-[14px] sm:grid-cols-2">
            {[
              ["LLM provider", health.llm_configured ? `${health.llm_provider} (${health.llm_model})` : `${health.llm_provider} — no key, using local fallback`],
              ["Embeddings", health.embedding_backend],
              ["Indexed chunks", String(health.vector_store_documents)],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between gap-12 border-b border-frost-gray/40 pb-6">
                <dt className="text-slate-gray">{label}</dt>
                <dd className="text-right text-ink-black">{value}</dd>
              </div>
            ))}
          </dl>
        </motion.div>
      )}

      {/* ── Latest summary ──────────────────────────── */}
      {!error && (
        <motion.div variants={fadeUp} className="space-y-12">
          <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
            Latest summary
          </p>
          <p className="whitespace-pre-wrap text-[16px] leading-[1.6] text-stone-gray">{recentSummary}</p>
        </motion.div>
      )}
    </motion.div>
  );
}
