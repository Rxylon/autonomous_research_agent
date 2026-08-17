"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { Badge } from "@/components/ui/badge";
import { ErrorState } from "@/components/ui/error-state";
import { fetchHistory } from "@/services/api";
import type { CriticMethod } from "@/types/research";

/** A heuristic score is not verification, so the list marks it rather than showing a bare number. */
const methodSuffix: Record<CriticMethod, string> = {
  llm: "",
  heuristic: " (heuristic)",
  empty: " (n/a)",
};

/* ── Animation variants ───────────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.15 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease } },
};

export default function HistoryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["history"],
    queryFn: fetchHistory,
    retry: false,
  });

  return (
    <div className="mx-auto max-w-3xl space-y-32">
      {/* ── Page heading ────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease }}
      >
        <h1 className="text-[24px] font-[500] leading-[1.33] text-ink-black">
          Research History
        </h1>
        <p className="mt-8 text-[16px] leading-[1.6] text-stone-gray">
          All prior runs in one place.
        </p>
      </motion.div>

      {error && <ErrorState error={error} what="research history" />}

      {/* ── History list ────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15, ease }}
        className="rounded-cards bg-surface-elevated shadow-xl-2 overflow-hidden"
        hidden={Boolean(error)}
      >
        {isLoading && (
          <div className="px-24 py-20 text-[14px] text-slate-gray">
            Loading research history…
          </div>
        )}

        {data?.length === 0 && !isLoading && (
          <div className="px-24 py-20 text-[14px] text-silver-mist">
            No research runs yet. Start one from the Chat page.
          </div>
        )}

        {data && data.length > 0 && (
          <motion.div variants={container} initial="hidden" animate="show">
            {data.map((item, index) => (
              <motion.div
                key={item.id}
                variants={fadeUp}
                whileHover={{ backgroundColor: "rgba(249, 249, 249, 0.5)" }}
                transition={{ duration: 0.15 }}
                className={`flex flex-col gap-8 px-24 py-20 sm:flex-row sm:items-center sm:justify-between ${
                  index > 0 ? "border-t border-frost-gray/40" : ""
                }`}
              >
                {/* Left: query + summary */}
                <div className="min-w-0 flex-1 space-y-4">
                  <div className="flex items-center gap-8">
                    <h3 className="truncate text-[15px] font-[500] text-ink-black">
                      {item.query}
                    </h3>
                    <Badge variant={item.status === "complete" ? "default" : "active"}>
                      {item.status}
                    </Badge>
                  </div>
                  <p className="truncate text-[13px] leading-[1.54] text-slate-gray">
                    {item.summary ?? "No summary available."}
                  </p>
                </div>

                {/* Right: date + critic score */}
                <div className="flex items-center gap-16 text-[13px] shrink-0">
                  <span className="text-silver-mist">
                    {new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                  </span>
                  <span className="font-[450] text-fire-orange">
                    {item.critic_score?.toFixed(2) ?? "—"}
                    <span className="text-silver-mist">
                      {methodSuffix[item.critic_method ?? "empty"]}
                    </span>
                  </span>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
