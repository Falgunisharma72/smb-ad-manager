"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Sparkles,
  ShieldCheck,
  Activity,
  Cpu,
  LineChart,
  Github,
  ExternalLink,
  Lock,
  TestTube2,
} from "lucide-react";
import { FadeIn, StaggerChildren, childFadeUp } from "@/components/fade-in";
import { AnimatedCounter } from "@/components/animated-counter";
import { useEffect, useState } from "react";
import { apiHealth } from "@/lib/api";

const STATS = [
  { label: "Reward functions", value: 5, subtitle: "independent, logged separately" },
  { label: "Anti-hack detectors", value: 5, subtitle: "live on each step" },
  { label: "Passing tests", value: 93, subtitle: "across 6 test files" },
  { label: "Bonus tracks", value: 3, subtitle: "Patronus + Scaler + Halluminate" },
];

const HIGHLIGHTS = [
  {
    title: "Live OpenEnv",
    body: "OpenEnv-compliant environment deployed to HuggingFace Spaces. Reset / step / state endpoints serve real simulation of the Meta Ads ecosystem.",
    icon: Activity,
    tint: "from-primary-soft to-accent-sky/70",
  },
  {
    title: "Reward-Hardened",
    body: "Five independent reward functions plus five anti-hack detectors — so judges can see attempted shortcuts being caught in real time.",
    icon: ShieldCheck,
    tint: "from-accent-peach/60 to-accent-rose/60",
  },
  {
    title: "Calibrated World",
    body: "User response, ad auction, and policy enforcement calibrated from WordStream India benchmarks + Meta Ad Library. Not a toy.",
    icon: TestTube2,
    tint: "from-accent-lavender/70 to-primary-soft",
  },
];

export default function DashboardPage() {
  const [health, setHealth] = useState<"loading" | "ok" | "down">("loading");

  useEffect(() => {
    apiHealth()
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"));
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero */}
      <FadeIn>
        <div className="relative overflow-hidden rounded-2xl glass-card p-8 md:p-12">
          <div className="relative z-10 max-w-3xl">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-white/60 px-3.5 py-1 text-xs font-medium text-muted-foreground mb-6"
            >
              <span className="relative flex h-2 w-2">
                <span
                  className={`absolute inline-flex h-full w-full rounded-full ${
                    health === "ok"
                      ? "bg-success animate-ping"
                      : health === "down"
                      ? "bg-danger"
                      : "bg-warning"
                  } opacity-60`}
                />
                <span
                  className={`relative inline-flex h-2 w-2 rounded-full ${
                    health === "ok"
                      ? "bg-success"
                      : health === "down"
                      ? "bg-danger"
                      : "bg-warning"
                  }`}
                />
              </span>
              {health === "ok"
                ? "Environment is live on HuggingFace Spaces"
                : health === "down"
                ? "Environment offline (build may be in progress)"
                : "Checking environment…"}
            </motion.div>

            <h1 className="font-serif text-5xl md:text-6xl leading-[1.05] tracking-tight mb-4">
              <span className="text-gradient">Reward-hardened RL</span>
              <br />
              for Meta Ads LLM agents.
            </h1>
            <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
              Small business owners lose thousands on Meta Ads because AI ad
              managers don&apos;t exist yet. We built the training ground that makes
              them possible.
            </p>

            <div className="flex flex-wrap gap-3">
              <Link href="/playground" className="btn btn-primary">
                <Cpu className="w-4 h-4" />
                Try the playground
              </Link>
              <a
                href="https://github.com/Falgunisharma72/smb-ad-manager"
                target="_blank"
                rel="noreferrer"
                className="btn btn-ghost border border-border"
              >
                <Github className="w-4 h-4" />
                Source
                <ExternalLink className="w-3.5 h-3.5 opacity-60" />
              </a>
            </div>
          </div>

          {/* Decorative floating blobs */}
          <motion.div
            aria-hidden
            animate={{ y: [0, -10, 0], rotate: [0, 4, 0] }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -top-10 -right-8 w-40 h-40 rounded-full bg-accent-peach/40 blur-2xl"
          />
          <motion.div
            aria-hidden
            animate={{ y: [0, 12, 0], rotate: [0, -5, 0] }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -bottom-14 left-1/3 w-48 h-48 rounded-full bg-accent-lavender/30 blur-3xl"
          />
        </div>
      </FadeIn>

      {/* Stat cards */}
      <section>
        <StaggerChildren className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((s) => (
            <motion.div
              key={s.label}
              variants={childFadeUp}
              whileHover={{ y: -3 }}
              className="rounded-2xl bg-white/80 border border-border p-5 shadow-soft hover:shadow-soft-md transition"
            >
              <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                {s.label}
              </div>
              <div className="font-serif text-4xl text-foreground">
                <AnimatedCounter value={s.value} />
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                {s.subtitle}
              </div>
            </motion.div>
          ))}
        </StaggerChildren>
      </section>

      {/* Highlight cards */}
      <section>
        <FadeIn delay={0.1}>
          <h2 className="font-serif text-3xl mb-6">Why this environment is different</h2>
        </FadeIn>
        <StaggerChildren className="grid md:grid-cols-3 gap-5">
          {HIGHLIGHTS.map((h) => {
            const Icon = h.icon;
            return (
              <motion.article
                key={h.title}
                variants={childFadeUp}
                whileHover={{ y: -4 }}
                className="rounded-2xl bg-white border border-border p-6 shadow-soft transition"
              >
                <div
                  className={`w-11 h-11 rounded-xl bg-gradient-to-br ${h.tint} flex items-center justify-center mb-5`}
                >
                  <Icon className="w-5 h-5 text-foreground/80" strokeWidth={1.8} />
                </div>
                <h3 className="font-semibold text-lg mb-2">{h.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {h.body}
                </p>
              </motion.article>
            );
          })}
        </StaggerChildren>
      </section>

      {/* Coming-soon banner for the other pages */}
      <FadeIn delay={0.2}>
        <div className="rounded-2xl border border-dashed border-border p-6 flex items-start gap-4 bg-white/40">
          <div className="w-10 h-10 rounded-xl bg-accent-lavender/40 flex items-center justify-center shrink-0">
            <Sparkles className="w-5 h-5 text-foreground/70" />
          </div>
          <div>
            <h3 className="font-medium mb-1">More coming in next build pass</h3>
            <p className="text-sm text-muted-foreground">
              Adversarial mode demo, training metrics dashboards, team page —
              currently under construction. The playground is fully functional.
            </p>
          </div>
        </div>
      </FadeIn>
    </div>
  );
}
