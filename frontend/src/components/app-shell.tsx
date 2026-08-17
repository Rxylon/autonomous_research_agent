"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, FileText, History, MessageSquare } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/history", label: "History", icon: History },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/dashboard", label: "Analytics", icon: BarChart3 },
];

/* ── Shared animation curves ──────────────────── */
const ease: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-surface-canvas">
      {/* ── Top navigation bar ──────────────────────── */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease }}
        className="sticky top-0 z-50 border-b border-cloud-canvas bg-surface-card/80 backdrop-blur-xl"
      >
        <div className="mx-auto flex h-[56px] max-w-[1200px] items-center justify-between px-16 md:px-24 lg:px-32">
          {/* Brand */}
          <Link href="/" className="flex items-center gap-10 shrink-0 group">
            <motion.div
              whileHover={{ scale: 1.08, rotate: -4 }}
              whileTap={{ scale: 0.95 }}
              transition={{ type: "spring", stiffness: 400, damping: 17 }}
              className="flex h-[28px] w-[28px] items-center justify-center rounded-lg bg-fire-orange"
            >
              <MessageSquare className="h-[14px] w-[14px] text-white" />
            </motion.div>
            <span className="text-[15px] font-[500] text-ink-black tracking-[-0.01em] transition-colors duration-200 group-hover:text-fire-orange">
              Research Assistant
            </span>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-4 relative">
            {navItems.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "relative flex items-center gap-6 rounded-menu-items px-12 py-8 text-[13px] font-[450] transition-colors duration-200",
                    isActive
                      ? "text-fire-orange"
                      : "text-ink-black/50 hover:text-ink-black",
                  )}
                >
                  {/* Active indicator — slides between nav items */}
                  {isActive && (
                    <motion.div
                      layoutId="nav-active-pill"
                      className="absolute inset-0 rounded-menu-items bg-fire-orange/8"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}

                  {/* Hover background — subtle scale-in */}
                  {!isActive && (
                    <span className="absolute inset-0 rounded-menu-items bg-cloud-canvas/0 transition-colors duration-200 hover:bg-cloud-canvas/50" />
                  )}

                  <motion.span
                    className="relative z-10 flex items-center gap-6"
                    initial={false}
                    animate={{ scale: 1 }}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    transition={{ type: "spring", stiffness: 400, damping: 20 }}
                  >
                    <Icon className="h-[14px] w-[14px]" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </motion.span>
                </Link>
              );
            })}
          </nav>
        </div>
      </motion.header>

      {/* ── Main content with page transitions ───── */}
      <AnimatePresence mode="wait">
        <motion.main
          key={pathname}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease }}
          className="mx-auto max-w-[1200px] px-16 py-32 md:px-24 md:py-40 lg:px-32 lg:py-48"
        >
          {children}
        </motion.main>
      </AnimatePresence>
    </div>
  );
}
