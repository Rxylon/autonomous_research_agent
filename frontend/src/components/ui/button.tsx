import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "ghost" | "nav";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-fire-orange text-white hover:brightness-110 active:brightness-95",
  ghost:
    "bg-transparent text-ink-black border border-cloud-canvas hover:bg-cloud-canvas/50 active:bg-cloud-canvas",
  nav:
    "bg-transparent text-ink-black/56 hover:text-ink-black px-[6px] py-[6px] rounded-none",
};

export function Button({ variant = "primary", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center font-[450] text-[16px] leading-[1.5] transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50",
        variant !== "nav" && "rounded-buttons px-[24px] py-[12px]",
        variantStyles[variant],
        className,
      )}
      {...props}
    />
  );
}
