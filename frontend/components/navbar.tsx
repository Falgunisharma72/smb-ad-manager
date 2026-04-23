"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, LayoutDashboard, Cpu, ShieldAlert, LineChart, LogOut, Info } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/playground", label: "Playground", icon: Cpu },
  { href: "/adversarial", label: "Adversarial", icon: ShieldAlert, disabled: true },
  { href: "/metrics", label: "Metrics", icon: LineChart, disabled: true },
  { href: "/about", label: "About", icon: Info, disabled: true },
];

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();

  async function onLogout() {
    await fetch("/api/logout", { method: "POST" });
    router.push("/login");
    router.refresh();
  }

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="sticky top-0 z-40 glass-card border-b border-border/60"
    >
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        {/* Brand */}
        <Link href="/dashboard" className="flex items-center gap-2.5 group">
          <motion.div
            whileHover={{ rotate: 20, scale: 1.1 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary-soft to-accent-lavender flex items-center justify-center"
          >
            <Sparkles className="w-4 h-4 text-primary" strokeWidth={2} />
          </motion.div>
          <span className="font-serif text-lg leading-none group-hover:text-primary transition">
            SMB Ad Manager
          </span>
        </Link>

        {/* Nav links */}
        <div className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.disabled ? "#" : item.href}
                aria-disabled={item.disabled}
                className={cn(
                  "relative px-3.5 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition",
                  active
                    ? "text-primary"
                    : "text-foreground/70 hover:text-foreground",
                  item.disabled && "opacity-40 cursor-not-allowed"
                )}
                onClick={(e) => item.disabled && e.preventDefault()}
              >
                <Icon className="w-4 h-4" strokeWidth={2} />
                {item.label}
                {active && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute inset-0 bg-primary-soft rounded-lg -z-10"
                    transition={{ type: "spring", stiffness: 250, damping: 28 }}
                  />
                )}
                {item.disabled && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                    soon
                  </span>
                )}
              </Link>
            );
          })}
        </div>

        <button
          onClick={onLogout}
          className="btn btn-ghost text-sm gap-1.5"
          title="Sign out"
        >
          <LogOut className="w-4 h-4" />
          <span className="hidden sm:inline">Sign out</span>
        </button>
      </div>
    </motion.nav>
  );
}
