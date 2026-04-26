"use client";

import Link from "next/link";
import {
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
import { FadeIn } from "@/components/fade-in";

const TEAM = [
  {
    name: "Falguni Sharma",
    role: "RL + Backend",
    affiliation: "Banasthali Vidyapith · CS-AI",
    note: "Machine Unlearning research · Reward design + agent training",
  },
  {
    name: "Sarthak Kala",
    role: "Environment + Eval",
    affiliation: "Chitkara University · CS-AI",
    note: "World simulation + tool API + scenario calibration",
  },
  {
    name: "Shrishty Kothiyal",
    role: "Pitch + Frontend",
    affiliation: "Banasthali Vidyapith · CS-AI",
    note: "Pitching the project to judges · founder demo storytelling · frontend showcase",
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
    note: "Group Relative Policy Optimization - RL without a critic.",
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
    note: "Targeted forgetting in fine-tuned models - informed our reward-isolation approach.",
  },
];

const BONUS_TRACKS = [
  {
    name: "Patronus AI",
    claim: "Schema-drift hardening",
    body:
      "Mid-episode policy drift (p6_health_disclaimer) injected at task tier ≥ medium. The agent must detect it via get_policy_updates and respond with rewrite_creative - exactly the failure mode Patronus benchmarks measure.",
  },
  {
    name: "Scaler AI Labs",
    claim: "Enterprise-grade reward isolation",
    body:
      "Five reward components logged separately (no aggregated black box). Five anti-hack detectors give judges visible proof that shortcuts are caught - the kind of governance an enterprise team would actually deploy.",
  },
  {
    name: "Halluminate",
    claim: "Multi-actor consistency",
    body:
      "User-response model + ad-auction + policy-enforcer act as three independent simulated actors. The agent's no-cheating reward (r5) verifies it only cites metrics it actually fetched - directly matches Halluminate's hallucination criterion.",
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-24">
      <FadeIn>
        <div className="max-w-4xl">
          <div className="text-sm uppercase tracking-[0.12em] text-primary font-medium mb-5">
            About
          </div>
          <h1 className="font-serif text-5xl md:text-6xl leading-[1.05] mb-6">
            <span className="text-gradient">Reward-hardened RL</span>
            <br />
            for the Indian SMB economy.
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            Built for the Scaler OpenEnv hackathon (April 2026). We don&apos;t
            ship an agent. We ship the environment that lets any agent learn
            what good Meta Ads management looks like, with guard-rails that
            catch shortcuts.
          </p>
        </div>
      </FadeIn>

      {/* Team */}
      <section>
        <FadeIn delay={0.05}>
          <div className="mb-10">
            <h2 className="font-serif text-4xl mb-3">Team</h2>
            <p className="text-lg text-muted-foreground">Three people, three lanes.</p>
          </div>
        </FadeIn>
        <div className="grid md:grid-cols-3 gap-5">
          {TEAM.map((t) => (
            <article
              key={t.name}
              className="card-surface p-7"
            >
              <div className="icon-chip w-12 h-12 mb-5">
                <span className="font-serif text-lg text-primary">
                  {t.name.charAt(0)}
                </span>
              </div>
              <div className="font-semibold text-xl">{t.name}</div>
              <div className="text-sm text-primary font-medium mt-1 mb-2">
                {t.role}
              </div>
              <div className="text-sm text-muted-foreground mb-4">
                {t.affiliation}
              </div>
              <p className="text-base text-muted-foreground leading-relaxed">
                {t.note}
              </p>
            </article>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section>
        <FadeIn delay={0.05}>
          <div className="mb-10">
            <h2 className="font-serif text-4xl mb-3">Architecture</h2>
            <p className="text-lg text-muted-foreground">Four layers, each independently testable.</p>
          </div>
        </FadeIn>
        <div className="grid md:grid-cols-2 gap-5">
          {STACK_LAYERS.map((l) => {
            const Icon = l.icon;
            return (
              <article
                key={l.title}
                className="card-surface p-7"
              >
                <div className="flex items-center gap-4 mb-5">
                  <span className="icon-chip w-11 h-11">
                    <Icon className="w-5 h-5" />
                  </span>
                  <h3 className="font-semibold text-xl">{l.title}</h3>
                </div>
                <ul className="space-y-3">
                  {l.items.map((it) => (
                    <li
                      key={it}
                      className="text-base text-muted-foreground flex gap-3 leading-relaxed"
                    >
                      <span className="mt-2.5 w-1.5 h-1.5 rounded-full bg-primary/80 shrink-0" />
                      <span>{it}</span>
                    </li>
                  ))}
                </ul>
              </article>
            );
          })}
        </div>
      </section>

      {/* Bonus tracks */}
      <section>
        <FadeIn delay={0.05}>
          <div className="mb-10">
            <h2 className="font-serif text-4xl mb-3">Bonus tracks claimed</h2>
            <p className="text-lg text-muted-foreground">
              Hooks the env exposes for each track.
            </p>
          </div>
        </FadeIn>
        <div className="grid md:grid-cols-3 gap-5">
          {BONUS_TRACKS.map((b) => (
            <article
              key={b.name}
              className="card-surface p-7"
            >
              <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-3">
                {b.name}
              </div>
              <h3 className="font-serif text-2xl mb-4">{b.claim}</h3>
              <p className="text-base text-muted-foreground leading-relaxed">
                {b.body}
              </p>
            </article>
          ))}
        </div>
      </section>

      {/* Citations */}
      <section>
        <FadeIn delay={0.05}>
          <div className="mb-10">
            <h2 className="font-serif text-4xl mb-3">Citations</h2>
            <p className="text-lg text-muted-foreground">
              The papers and specs this work builds on.
            </p>
          </div>
        </FadeIn>
        <div className="card-surface divide-y divide-border overflow-hidden">
          {CITATIONS.map((c) => (
            <div
              key={c.title}
              className="px-7 py-6 flex items-start gap-5"
            >
              <span className="icon-chip shrink-0 w-11 h-11">
                <BookOpen className="w-5 h-5" />
              </span>
              <div className="flex-1">
                <div className="font-semibold text-lg">{c.title}</div>
                <div className="text-sm text-muted-foreground font-mono mb-2">
                  {c.venue}
                </div>
                <p className="text-base text-muted-foreground leading-relaxed">
                  {c.note}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <FadeIn delay={0.05}>
        <div className="card-surface p-10 md:p-12 flex flex-col md:flex-row md:items-center gap-6">
          <div className="flex-1">
            <h3 className="font-serif text-3xl mb-3">Try it yourself</h3>
            <p className="text-lg text-muted-foreground">
              The environment is live, the trained model is on HF Hub, the code
              is open.
            </p>
          </div>
          <div className="flex gap-3 flex-wrap">
            <Link href="/founder" className="btn btn-primary text-base px-5 py-2.5">
              <Sparkles className="w-4 h-4" />
              Founder mode
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <a
              href="https://github.com/Falgunisharma72/smb-ad-manager"
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost border border-border text-base px-5 py-2.5"
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
