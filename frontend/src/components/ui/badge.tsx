import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type BadgeVariant = "default" | "active";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: "bg-pale-sienna/40 text-fire-orange border-pale-sienna",
  active: "bg-fire-orange text-white border-fire-orange",
};

export function Badge({ variant = "default", className, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-tags border px-[12px] py-[4px] text-[12px] font-[450] leading-[1.33]",
        variantStyles[variant],
        className,
      )}
      {...props}
    />
  );
}
