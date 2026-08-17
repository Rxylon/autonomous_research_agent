"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { fetchHistory, reportDownloadUrl } from "@/services/api";

/* ── Animation config ─────────────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease } },
};

export default function ReportsPage() {
  const { data, error } = useQuery({ queryKey: ["history"], queryFn: fetchHistory, retry: false });
  const latest = data?.find((item) => item.report) ?? data?.[0];

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="mx-auto max-w-3xl space-y-32"
    >
      {/* ── Page heading ────────────────────────────── */}
      <motion.div variants={fadeUp} className="flex items-end justify-between gap-16">
        <div>
          <h1 className="text-[24px] font-[500] leading-[1.33] text-ink-black">Reports</h1>
          <p className="mt-8 text-[16px] leading-[1.6] text-stone-gray">
            Structured, source-backed research outputs.
          </p>
        </div>
        {latest?.id && (
          <motion.div
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <Link href={`/reports/detail?id=${latest.id}`}>
              <Button variant="ghost" className="text-[14px] px-16 py-8">
                View detail
              </Button>
            </Link>
          </motion.div>
        )}
      </motion.div>

      {error && <ErrorState error={error} what="reports" />}

      {/* ── Latest report ───────────────────────────── */}
      {!error && (
        <motion.div
          variants={fadeUp}
          className="rounded-cards bg-surface-elevated p-24 shadow-xl-2 md:p-32 space-y-16"
        >
          <div className="flex flex-wrap items-start justify-between gap-16">
            <div>
              <p className="text-[10px] font-[500] uppercase tracking-[0.1em] text-fire-orange">
                Latest report
              </p>
              <h2 className="mt-8 text-[20px] font-[500] leading-[1.43] text-ink-black">
                {latest?.query ?? "No report yet"}
              </h2>
            </div>

            {latest?.report && (
              <div className="flex items-center gap-8">
                {(["md", "json", "pdf"] as const).map((format) => (
                  <a
                    key={format}
                    href={reportDownloadUrl(latest.id, format)}
                    className="inline-flex items-center gap-4 rounded-menu-items border border-cloud-canvas px-10 py-6 text-[12px] font-[450] text-stone-gray transition-colors hover:border-fire-orange hover:text-fire-orange"
                  >
                    <Download className="h-[12px] w-[12px]" />
                    {format.toUpperCase()}
                  </a>
                ))}
              </div>
            )}
          </div>

          <pre className="overflow-auto whitespace-pre-wrap rounded-xl border border-cloud-canvas bg-surface-card p-20 font-[family-name:var(--font-geistmono)] text-[13px] leading-[1.6] text-stone-gray">
            {latest?.report?.markdown ?? latest?.summary ?? "Run a query to generate a structured report."}
          </pre>

          {latest?.report && (
            <p className="text-[12px] leading-[1.5] text-silver-mist">
              The Markdown and JSON downloads are served from stored run history. The PDF is read
              from the backend&apos;s local filesystem, which is ephemeral on free hosting tiers — it
              may return 410 after a restart.
            </p>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}
