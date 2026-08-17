import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type CardVariant = "base" | "elevated" | "code";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
}

const variantStyles: Record<CardVariant, string> = {
  base: "bg-surface-card shadow-xl",
  elevated: "bg-surface-elevated shadow-xl-2",
  code: "bg-surface-card rounded-[20px] shadow-xl-3",
};

export function Card({ variant = "base", className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-cards p-16",
        variantStyles[variant],
        className,
      )}
      {...props}
    />
  );
}
