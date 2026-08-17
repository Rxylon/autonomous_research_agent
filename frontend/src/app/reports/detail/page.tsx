"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowLeft, Download } from "lucide-react";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";

import { ErrorState } from "@/components/ui/error-state";
import { fetchRun, reportDownloadUrl } from "@/services/api";

/* ── Animation config ─────────────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease } },
};

function ReportDetailContent() {
  const searchParams = useSearchParams();
  const id = searchParams.get("id");
  const { data, error, isLoading } = useQuery({
    queryKey: ["run", id],
    queryFn: () => fetchRun(id as string),
    enabled: Boolean(id),
    retry: false,
  });

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="mx-auto max-w-3xl space-y-32"
    >
      {/* ── Back link + heading ─────────────────────── */}
      <motion.div variants={fadeUp} className="space-y-16">
        <Link
          href="/reports"
          className="group inline-flex items-center gap-6 text-[13px] font-[450] text-silver-mist transition-colors duration-200 hover:text-ink-black"
        >
          <motion.span
            className="inline-block"
            whileHover={{ x: -3 }}
            transition={{ type: "spring", stiffness: 400, damping: 20 }}
          >
            <ArrowLeft className="h-[14px] w-[14px]" />
          </motion.span>
          Back to reports
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-16">
          <div>
            <h1 className="text-[24px] font-[500] leading-[1.33] text-ink-black">
              {data?.query ?? (id ? `Report ${id}` : "No report selected")}
            </h1>
            {data?.summary && (
              <p className="mt-8 whitespace-pre-wrap text-[16px] leading-[1.6] text-stone-gray">
                {data.summary}
              </p>
            )}
          </div>

          {data?.report && id && (
            <div className="flex items-center gap-8">
              {(["md", "json", "pdf"] as const).map((format) => (
                <a
                  key={format}
                  href={reportDownloadUrl(id, format)}
                  className="inline-flex items-center gap-4 rounded-menu-items border border-cloud-canvas px-10 py-6 text-[12px] font-[450] text-stone-gray transition-colors hover:border-fire-orange hover:text-fire-orange"
                >
                  <Download className="h-[12px] w-[12px]" />
                  {format.toUpperCase()}
                </a>
              ))}
            </div>
          )}
        </div>
      </motion.div>

      {!id && (
        <p className="text-[14px] text-silver-mist">
          No run id was supplied. Pick a report from the{" "}
          <Link href="/reports" className="underline">
            reports list
          </Link>
          .
        </p>
      )}

      {error && <ErrorState error={error} what="this report" />}

      {/* ── Report content ──────────────────────────── */}
      {!error && id && (
        <motion.div
          variants={fadeUp}
          className="rounded-cards bg-surface-elevated p-24 shadow-xl-2 md:p-32"
        >
          <pre className="overflow-auto whitespace-pre-wrap font-[family-name:var(--font-geistmono)] text-[13px] leading-[1.6] text-stone-gray">
            {isLoading
              ? "Loading report…"
              : (data?.report?.markdown ??
                data?.summary ??
                "No report content has been stored for this run yet.")}
          </pre>
        </motion.div>
      )}
    </motion.div>
  );
}

export default function ReportDetailPage() {
  return (
    <Suspense fallback={<div className="text-[14px] text-silver-mist">Loading…</div>}>
      <ReportDetailContent />
    </Suspense>
  );
}
