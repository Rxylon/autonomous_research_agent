import { cn } from "@/lib/utils";
import type { TextareaHTMLAttributes } from "react";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  size?: "default" | "large";
}

export function Textarea({ className, size = "default", ...props }: TextareaProps) {
  return (
    <textarea
      className={cn(
        "w-full rounded-inputs border border-cloud-canvas bg-surface-elevated px-16 py-12 text-ink-black placeholder:text-silver-mist outline-none transition-all duration-200 focus:border-fire-orange focus:ring-2 focus:ring-fire-orange/15 resize-y",
        size === "large"
          ? "min-h-[140px] text-[16px] leading-[1.6]"
          : "min-h-[100px] text-[14px] leading-[1.54]",
        className,
      )}
      {...props}
    />
  );
}
