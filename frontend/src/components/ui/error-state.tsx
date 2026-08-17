"use client";

import { AlertTriangle } from "lucide-react";

import { backendUrl } from "@/services/api";

/**
 * Shown wherever a backend read fails.
 *
 * Every page needs this because the API client no longer invents data on failure —
 * it used to return a fabricated run with a made-up source and critic score, which
 * meant a dead backend looked identical to a successful research run.
 */
export function ErrorState({ error, what }: { error: unknown; what: string }) {
  const message = error instanceof Error ? error.message : String(error);

  return (
    <div className="flex items-start gap-10 rounded-cards border border-red-300/60 bg-red-50 p-20 text-[13px] leading-[1.54] text-red-900">
      <AlertTriangle className="mt-[2px] h-[15px] w-[15px] shrink-0" />
      <div className="space-y-4">
        <p>
          <strong className="font-[500]">Could not load {what}.</strong> {message}
        </p>
        <p className="text-red-800/80">
          Backend: <code className="font-[family-name:var(--font-geistmono)]">{backendUrl}</code>
        </p>
      </div>
    </div>
  );
}
