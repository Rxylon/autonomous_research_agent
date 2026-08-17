"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type StepStatus = "pending" | "running" | "complete";

interface Step {
  stage: string;
  status: StepStatus;
  message: string;
}

interface StepperProps {
  steps: Step[];
  className?: string;
}

const statusDot: Record<StepStatus, string> = {
  pending: "bg-frost-gray",
  running: "bg-fire-orange",
  complete: "bg-emerald-500",
};

const statusLabel: Record<StepStatus, string> = {
  pending: "text-silver-mist",
  running: "text-fire-orange",
  complete: "text-stone-gray",
};

const statusLine: Record<StepStatus, string> = {
  pending: "bg-frost-gray",
  running: "bg-fire-orange/40",
  complete: "bg-emerald-500/40",
};

export function Stepper({ steps, className }: StepperProps) {
  return (
    <div className={cn("flex items-start gap-0 w-full", className)}>
      {steps.map((step, index) => {
        const isLast = index === steps.length - 1;
        const isRunning = step.status === "running";

        return (
          <div key={step.stage} className="flex items-start flex-1 min-w-0">
            {/* Step node */}
            <div className="flex flex-col items-center gap-8 shrink-0">
              {/* Dot — animates on status change */}
              <motion.div
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{
                  scale: isRunning ? [1, 1.3, 1] : 1,
                  opacity: 1,
                }}
                transition={
                  isRunning
                    ? { scale: { repeat: Infinity, duration: 1.8, ease: "easeInOut" }, opacity: { duration: 0.3 } }
                    : { type: "spring", stiffness: 500, damping: 25 }
                }
                className={cn(
                  "h-[10px] w-[10px] rounded-full transition-colors duration-300",
                  statusDot[step.status],
                  isRunning && "shadow-[0_0_0_4px_rgba(255,77,0,0.15)]",
                )}
              />
              {/* Label */}
              <motion.p
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.06 + 0.1, duration: 0.3 }}
                className={cn(
                  "text-[11px] font-[500] uppercase tracking-[0.08em] leading-[1] whitespace-nowrap",
                  statusLabel[step.status],
                )}
              >
                {step.stage}
              </motion.p>
            </div>

            {/* Connector line — grows from left to right */}
            {!isLast && (
              <div className="flex-1 flex items-start pt-[4px] px-4 min-w-[16px]">
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: index * 0.08 + 0.05, duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                  style={{ originX: 0 }}
                  className={cn(
                    "h-[2px] w-full rounded-full transition-colors duration-300",
                    statusLine[steps[index + 1]?.status === "pending" ? "pending" : step.status],
                  )}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
