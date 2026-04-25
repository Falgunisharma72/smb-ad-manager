"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  GraduationCap,
  Code2,
  Sparkles,
  ExternalLink,
  Github,
  BookOpen,
  Layers,
  Activity,
  ShieldAlert,
  Cpu,
  ArrowRight,
} from "lucide-react";
import { FadeIn, StaggerChildren, childFadeUp } from "@/components/fade-in";

const TEAM = [
  {
    name: "Falguni Sharma",
    role: "RL + Backend",
    affiliation: "Banasthali Vidyapith · CS-AI",
    note: "Machine Unlearning research · Reward design + agent training",
  },
  {
    name: "Sarthak",
    role: "Environment + Eval",
    affiliation: "—",
    note: "World simulation + tool API + scenario calibration",
  },
  {
    name: "Shrishty",
    role: "Frontend + Pitch",
    affiliation: "—",
    note: "Founder UX + judge demo flow + storytelling",
  },
];

const STACK_LAYERS = [
  {
    title: "Environment",
    icon: Activity,
    items: [
      "OpenEnv-compliant /reset · /step · /state · /healthz",
      "Pydantic 2 strict schemas (15 models · extra='forbid')",
      "8 mock Marketing API tools dispatched per step",
      "FastAPI on HuggingFace Spaces (Docker, 2vCPU/16GB)",
    ],
    tint: "from-primary-soft to-accent-sky/60",
  },
  {
    title: "World simulation",
    icon: Layers,
    items: [
      "User-response model calibrated from WordStream India + Meta Ad Library",
      "3 industries × 10 SMB profiles (skincare / food / fitness)",
      "Policy enforcer with 5 always-on rules + mid-episode drift",
      "Auction-style ad serving with creative-quality penalties",
    ],
    tint: "from-accent-peach/60 to-accent-rose/60",
  },
  {
    title: "Reward stack",
    icon: ShieldAlert,
    items: [
      "5 independent reward functions logged separately",
      "5 anti-hack detectors run live on every step",
      "Multiplicative kill-switch on policy violations (r2 = 0 → score = 0)",
      "Partial-credit shaping for malformed JSON (no zero-signal failures)",
    ],
    tint: "from-accent-lavender/60 to-primary-soft",
  },
  {
    title: "Training",
    icon: Cpu,
    items: [
      "Qwen 2.5 1.5B Instruct base (no license gate)",
      "100 hand-crafted SFT examples · LoRA r=16 · 3 epochs",
      "GRPO with group size 2 (2-GRPO, arxiv 2510.00977)",
      "200 steps · ~1h on Colab Pro L4 · W&B logged",
    ],
    tint: "from-primary-soft to-accent-lavender/70",
  },
];

const CITATIONS = [
  {
    title: "DeepSeek-R1 (GRPO origin)",
    venue: "arxiv 2501.12948",
    note: "Group Relative Policy Optimization — RL without a critic.",
  },
  {
    title: "2-GRPO: compute-efficient RL",
    venue: "arxiv 2510.00977",
    note: "Group size 2 cuts rollouts by 8× with negligible final-score loss.",
  },
  {
    title: "OpenEnv specification",
    venue: "huggingface/openenv",
    note: "Reset / step / state contract for RL environments.",
  },
  {
    title: "Machine Unlearning (Falguni's prior research)",
    venue: "Banasthali Vidyapith · 2025",
    note: "Targeted forgetting in fine-tuned models — informed our reward-isolation approach.",
  },
];

const BONUS_TRACKS = [
  {
    name: "Patronus AI",
    claim: "Schema-drift hardening",
    body:
      "Mid-episode policy drift (p6_health_disclaimer) injected at task tier ≥ medium. The agent must detect it via get_policy_updates and respond with rewrite_creative — exactly the failure mode Patronus benchmarks measure.",
  },
  {
    name: "Scaler AI Labs",
    claim: "Enterprise-grade reward isolation",
    body:
      "Five reward components logged separately (no aggregated black box). Five anti-hack detectors give judges visible proof that shortcuts are caught — the kind of governance an enterprise team would actually deploy.",
  },
  {
    name: "Halluminate",
    claim: "Multi-actor consistency",
    body:
      "User-response model + ad-auction + policy-enforcer act as three independent simulated actors. The agent's no-cheating reward (r5) verifies it only cites metrics it actually fetched — directly matches Halluminate's hallucination criterion.",
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-12">
      <FadeIn>
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-white/60 px-3 py-1 text-xs font-medium text-muted-foreground mb-3">
            <Sparkles className="w-3 h-3 text-primary" />
            About
          </div>
          <h1 className="font-serif text-4xl md:text-5xl mb-3">
            <span className="text-gradient">Reward-hardened RL</span> for the
            Indian SMB economy.
          </h1>
          <p className="text-muted-foreground max-w-3xl text-lg leading-relaxed">
            Built for the Scaler OpenEnv hackathon (April 2026). We don't ship
            an agent — we ship the environment that lets <i>any</i> agent
            actually learn what good Meta Ads management looks like, with
            guard-rails that catch the shortcuts.
          </p>
        </div>
      </FadeIn>

      {/* Team */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-5">Team</h2>
        </FadeIn>
        <StaggerChildren className="grid md:grid-cols-3 gap-4">
          {TEAM.map((t) => (
            <motion.div
              key={t.name}
              variants={childFadeUp}
              className="rounded-2xl bg-white border border-border p-5 shadow-soft"
            >
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary-soft to-accent-lavender flex items-center justify-center mb-4">
                <GraduationCap className="w-5 h-5 text-foreground/80" />
              </div>
              <div className="font-medium">{t.name}</div>
              <div className="text-xs text-primary font-medium mb-1">
                {t.role}
              </div>
              <div className="text-xs text-muted-foreground mb-2">
                {t.affiliation}
              </div>
              <p className="text-sm text-foreground/70 leading-snug">
                {t.note}
              </p>
            </motion.div>
          ))}
        </StaggerChildren>
      </section>

      {/* Architecture */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-5">Architecture</h2>
        </FadeIn>
        <StaggerChildren className="grid md:grid-cols-2 gap-4">
          {STACK_LAYERS.map((l) => {
            const Icon = l.icon;
            return (
              <motion.article
                key={l.title}
                variants={childFadeUp}
                className="rounded-2xl bg-white border border-border p-6 shadow-soft"
              >
                <div
                  className={`w-11 h-11 rounded-xl bg-gradient-to-br ${l.tint} flex items-center justify-center mb-4`}
                >
                  <Icon className="w-5 h-5 text-foreground/80" />
                </div>
                <h3 className="font-semibold text-lg mb-3">{l.title}</h3>
                <ul className="space-y-1.5">
                  {l.items.map((it) => (
                    <li
                      key={it}
                      className="text-sm text-foreground/75 flex gap-2 leading-snug"
                    >
                      <span className="mt-1.5 w-1 h-1 rounded-full bg-primary shrink-0" />
                      <span>{it}</span>
                    </li>
                  ))}
                </ul>
              </motion.article>
            );
          })}
        </StaggerChildren>
      </section>

      {/* Bonus tracks */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-5">Bonus tracks claimed</h2>
        </FadeIn>
        <StaggerChildren className="grid md:grid-cols-3 gap-4">
          {BONUS_TRACKS.map((b) => (
            <motion.div
              key={b.name}
              variants={childFadeUp}
              className="rounded-2xl bg-gradient-to-br from-white to-muted/30 border border-border p-5 shadow-soft"
            >
              <div className="text-xs uppercase tracking-wider text-primary font-semibold mb-2">
                {b.name}
              </div>
              <h3 className="font-serif text-xl mb-2">{b.claim}</h3>
              <p className="text-sm text-foreground/70 leading-relaxed">
                {b.body}
              </p>
            </motion.div>
          ))}
        </StaggerChildren>
      </section>

      {/* Citations */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-5 flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-primary" /> Citations
          </h2>
        </FadeIn>
        <div className="rounded-2xl bg-white border border-border divide-y divide-border overflow-hidden shadow-soft">
          {CITATIONS.map((c) => (
            <div
              key={c.title}
              className="p-4 flex items-start gap-4 hover:bg-muted/20 transition"
            >
              <div className="w-9 h-9 rounded-lg bg-primary-soft/60 flex items-center justify-center shrink-0">
                <BookOpen className="w-4 h-4 text-primary" />
              </div>
              <div className="flex-1">
                <div className="font-medium">{c.title}</div>
                <div className="text-xs text-muted-foreground font-mono mb-1">
                  {c.venue}
                </div>
                <p className="text-sm text-foreground/70">{c.note}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <FadeIn delay={0.1}>
        <div className="rounded-2xl glass-card p-8 flex flex-col md:flex-row items-start md:items-center gap-5">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-soft to-accent-peach/70 flex items-center justify-center shrink-0">
            <Code2 className="w-6 h-6 text-foreground/80" />
          </div>
          <div className="flex-1">
            <h3 className="font-serif text-2xl mb-1">Try it yourself</h3>
            <p className="text-sm text-muted-foreground">
              The environment is live, the trained model is on HF Hub, the code
              is open.
            </p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Link href="/founder" className="btn btn-primary">
              <Sparkles className="w-4 h-4" />
              Founder mode
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <a
              href="https://github.com/Falgunisharma72/smb-ad-manager"
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost border border-border"
            >
              <Github className="w-4 h-4" />
              GitHub
              <ExternalLink className="w-3.5 h-3.5 opacity-60" />
            </a>
          </div>
        </div>
      </FadeIn>
    </div>
  );
}
