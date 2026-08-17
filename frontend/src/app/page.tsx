"use client";

import { motion } from "framer-motion";
import { ResearchConsole } from "@/components/research-console";

/* ── Stagger animation variants ───────────────── */
const container = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.12, delayChildren: 0.1 },
  },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl space-y-40">
      {/* ── Hero heading ────────────────────────────── */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-12 text-center"
      >
        <motion.h1
          variants={fadeUp}
          className="text-[40px] font-[500] leading-[1.1] tracking-[-0.02em] text-ink-black md:text-[52px] md:leading-[1.07] md:tracking-[-0.52px]"
        >
          Research that reads, reasons, and reports.
        </motion.h1>
        <motion.p
          variants={fadeUp}
          className="mx-auto max-w-xl text-[16px] leading-[1.6] text-stone-gray"
        >
          Turn a single prompt into a multi-agent research workflow with planning, retrieval, summarization, and hallucination checks.
        </motion.p>
      </motion.div>

      {/* ── Research console ────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        <ResearchConsole />
      </motion.div>
    </div>
  );
}
