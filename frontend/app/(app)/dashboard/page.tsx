"use client";

import Image from "next/image";
import Link from "next/link";
import {
  Sparkles,
  ShieldCheck,
  Activity,
  Cpu,
  Github,
  ExternalLink,
  TestTube2,
  Target,
  ShieldAlert,
  CheckCircle2,
  Trophy,
} from "lucide-react";
import { FadeIn } from "@/components/fade-in";
import { useEffect, useState } from "react";
import { apiHealth } from "@/lib/api";

const STATS = [
  {
    label: "Reward functions",
    value: 5,
    subtitle: "independent, logged separately",
    icon: Target,
  },
  {
    label: "Anti-hack detectors",
    value: 5,
    subtitle: "live on each step",
    icon: ShieldAlert,
  },
  {
    label: "Passing tests",
    value: 93,
    subtitle: "across 6 test files",
    icon: CheckCircle2,
  },
  {
    label: "Bonus tracks",
    value: 3,
    subtitle: "Patronus + Scaler + Halluminate",
    icon: Trophy,
  },
];

const HIGHLIGHTS = [
  {
    title: "Live OpenEnv",
    body: "OpenEnv-compliant environment deployed to HuggingFace Spaces. Reset / step / state endpoints serve real simulation of the Meta Ads ecosystem.",
    icon: Activity,
  },
  {
    title: "Reward-Hardened",
    body: "Five independent reward functions plus five anti-hack detectors, so judges can see attempted shortcuts being caught in real time.",
    icon: ShieldCheck,
  },
  {
    title: "Calibrated World",
    body: "User response, ad auction, and policy enforcement calibrated from WordStream India benchmarks and Meta Ad Library.",
    icon: TestTube2,
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
    <div className="space-y-24">
      {/* Hero - split layout with chart preview on the right */}
      <FadeIn>
        <div className="grid lg:grid-cols-[1.1fr_1fr] gap-10 lg:gap-16 items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/40 px-3.5 py-1.5 text-sm font-medium text-muted-foreground mb-8">
              <span
                className={`inline-flex h-2 w-2 rounded-full ${
                  health === "ok"
                    ? "bg-success"
                    : health === "down"
                    ? "bg-danger"
                    : "bg-warning"
                }`}
              />
              {health === "ok"
                ? "Environment live on HuggingFace Spaces"
                : health === "down"
                ? "Environment offline"
                : "Checking environment"}
            </div>

            <h1 className="font-serif text-5xl md:text-7xl leading-[1.02] tracking-tight mb-7">
              <span className="text-gradient">Reward-hardened RL</span>
              <br />
              for Meta Ads LLM agents.
            </h1>
            <p className="text-xl text-muted-foreground mb-10 leading-relaxed">
              Small business owners lose thousands on Meta Ads because reliable
              AI ad managers don&apos;t exist yet. We built the training
              environment that makes them possible.
            </p>

            <div className="flex flex-wrap gap-3">
              <Link href="/founder" className="btn btn-primary text-base px-5 py-2.5">
                <Sparkles className="w-4 h-4" />
                Try Founder Mode
              </Link>
              <Link href="/playground" className="btn btn-ghost border border-border text-base px-5 py-2.5">
                <Cpu className="w-4 h-4" />
                Researcher playground
              </Link>
              <a
                href="https://github.com/Falgunisharma72/smb-ad-manager"
                target="_blank"
                rel="noreferrer"
                className="btn btn-ghost border border-border text-base px-5 py-2.5"
              >
                <Github className="w-4 h-4" />
                Source
                <ExternalLink className="w-3.5 h-3.5 opacity-60" />
              </a>
            </div>
          </div>

          {/* Hero image - the actual training reward curve as social proof */}
          <div className="card-surface p-6 lg:p-7">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-1">
                  Training run · Qwen 2.5 1.5B + GRPO
                </div>
                <div className="font-serif text-2xl">Reward 0.41 → 0.71</div>
              </div>
              <div className="text-right">
                <div className="font-serif text-3xl text-primary tabular-nums">+73%</div>
                <div className="text-xs text-muted-foreground">over SFT baseline</div>
              </div>
            </div>
            <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
              <Image
                src="/charts/grpo_1_5b_reward.png"
                alt="GRPO 1.5B mean reward over 200 training steps"
                width={1100}
                height={620}
                className="w-full h-auto"
                priority
              />
            </div>
            <div className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Mean reward across 200 GRPO steps. 51 minutes on a single Colab
              L4. Public W&amp;B run available below.
            </div>
          </div>
        </div>
      </FadeIn>

      {/* Stat highlight cards - the headline numbers, treated like trophies */}
      <section>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          {STATS.map((s) => {
            const Icon = s.icon;
            return (
              <article
                key={s.label}
                className="stat-card group"
              >
                <span className="icon-chip w-11 h-11 mb-6 group-hover:bg-[hsl(var(--primary)/0.18)] transition-colors">
                  <Icon className="w-5 h-5" strokeWidth={1.8} />
                </span>
                <div className="font-serif text-7xl leading-none text-primary tabular-nums tracking-tight">
                  {s.value}
                </div>
                <div className="text-base font-semibold text-foreground mt-5">
                  {s.label}
                </div>
                <div className="text-sm text-muted-foreground mt-1.5 leading-relaxed">
                  {s.subtitle}
                </div>
              </article>
            );
          })}
        </div>
      </section>

      {/* Highlight cards */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-4xl mb-3">Why this environment is different</h2>
          <p className="text-lg text-muted-foreground mb-10 max-w-2xl">
            Three design decisions that separate it from a toy gym.
          </p>
        </FadeIn>
        <div className="grid md:grid-cols-3 gap-5">
          {HIGHLIGHTS.map((h) => {
            const Icon = h.icon;
            return (
              <article
                key={h.title}
                className="card-surface p-7"
              >
                <span className="icon-chip mb-6 w-11 h-11">
                  <Icon className="w-5 h-5" strokeWidth={1.8} />
                </span>
                <h3 className="font-semibold text-xl mb-3">{h.title}</h3>
                <p className="text-base text-muted-foreground leading-relaxed">
                  {h.body}
                </p>
              </article>
            );
          })}
        </div>
      </section>

      {/* Baseline-vs-trained comparison - direct hit on the "show improvement" criterion */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-4xl mb-3">Baselines vs trained</h2>
          <p className="text-lg text-muted-foreground mb-10 max-w-2xl">
            Random and untrained-base sit at the floor (env returns 0 for
            invalid actions). SFT is a real but small lift. SFT + GRPO is
            where the env actually rewards skill.
          </p>
        </FadeIn>
        <div className="card-surface p-6 lg:p-8">
          <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
            <Image
              src="/charts/baseline_comparison.png"
              alt="Reward bar chart: random 0.05, untrained 0.05, +SFT 0.41, +SFT+GRPO 0.71"
              width={1600}
              height={880}
              className="w-full h-auto"
            />
          </div>
          <div className="grid sm:grid-cols-4 gap-4 mt-6 text-center">
            <div>
              <div className="font-serif text-3xl text-muted-foreground tabular-nums">0.05</div>
              <div className="text-xs uppercase tracking-[0.08em] text-muted-foreground mt-1">Random</div>
            </div>
            <div>
              <div className="font-serif text-3xl text-muted-foreground tabular-nums">0.05</div>
              <div className="text-xs uppercase tracking-[0.08em] text-muted-foreground mt-1">Untrained 1.5B</div>
            </div>
            <div>
              <div className="font-serif text-3xl text-foreground tabular-nums">0.41</div>
              <div className="text-xs uppercase tracking-[0.08em] text-muted-foreground mt-1">+ SFT</div>
            </div>
            <div>
              <div className="font-serif text-3xl text-primary tabular-nums">0.71</div>
              <div className="text-xs uppercase tracking-[0.08em] text-primary mt-1 font-semibold">+ SFT + GRPO</div>
            </div>
          </div>
        </div>
      </section>

      {/* Visual proof strip - all four training charts side by side */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-4xl mb-3">The training, visualised</h2>
          <p className="text-lg text-muted-foreground mb-10 max-w-2xl">
            Real plots from the W&amp;B runs. Loss curves, reward dynamics, the
            scaling-study failure. No mockups.
          </p>
        </FadeIn>
        <div className="grid md:grid-cols-2 gap-5">
          <article className="card-surface p-6 lg:p-7">
            <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-2">
              Stage 1 · SFT
            </div>
            <h3 className="font-serif text-2xl mb-4">Loss 2.32 → 0.165</h3>
            <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
              <Image
                src="/charts/sft_loss_curve.png"
                alt="SFT loss + token accuracy across 3 epochs"
                width={1100}
                height={520}
                className="w-full h-auto"
              />
            </div>
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Token accuracy climbed 57% to 95.6% across 3 epochs.
            </p>
          </article>

          <article className="card-surface p-6 lg:p-7">
            <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-2">
              Stage 2 · GRPO 1.5B
            </div>
            <h3 className="font-serif text-2xl mb-4">Reward 0.41 → 0.71</h3>
            <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
              <Image
                src="/charts/grpo_1_5b_reward.png"
                alt="1.5B GRPO mean reward across 200 steps"
                width={1100}
                height={620}
                className="w-full h-auto"
              />
            </div>
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Mean reward over 200 steps. Healthy variance throughout.
            </p>
          </article>

          <article className="card-surface p-6 lg:p-7">
            <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-2">
              Reward variance
            </div>
            <h3 className="font-serif text-2xl mb-4">Non-zero std every step</h3>
            <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
              <Image
                src="/charts/1.png"
                alt="Reward std across 200 GRPO steps"
                width={1100}
                height={520}
                className="w-full h-auto"
              />
            </div>
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Why GRPO worked: every batch carried real learning signal.
            </p>
          </article>

          <article className="card-surface p-6 lg:p-7">
            <div className="text-xs uppercase tracking-[0.1em] text-primary font-medium mb-2">
              Scaling study · 3B v2
            </div>
            <h3 className="font-serif text-2xl mb-4">Plateau at 0.35</h3>
            <div className="rounded-lg overflow-hidden border border-border/60 bg-background/40">
              <Image
                src="/charts/grpo_3b_v2_flat.png"
                alt="3B v2 GRPO reward stuck at 0.35"
                width={1100}
                height={620}
                className="w-full h-auto"
              />
            </div>
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Documented research finding: capacity-driven over-generalization.
            </p>
          </article>
        </div>
      </section>

      {/* Site map */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-4xl mb-3">Explore</h2>
          <p className="text-lg text-muted-foreground mb-10 max-w-2xl">
            Four entry points into the project, depending on whether you&apos;re
            judging, building, or just curious.
          </p>
        </FadeIn>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link href="/founder" className="card-surface p-6 group">
            <div className="text-xs uppercase tracking-[0.08em] text-primary font-medium mb-3">Founder Mode</div>
            <div className="font-medium text-lg mb-2">Try the AI Ad Manager</div>
            <div className="text-sm text-muted-foreground leading-relaxed">Fill a brief, watch a 7-day plan reveal.</div>
          </Link>
          <Link href="/adversarial" className="card-surface p-6 group">
            <div className="text-xs uppercase tracking-[0.08em] text-primary font-medium mb-3">Adversarial</div>
            <div className="font-medium text-lg mb-2">Watch hacks get caught</div>
            <div className="text-sm text-muted-foreground leading-relaxed">5 detectors firing live on attacker prompts.</div>
          </Link>
          <Link href="/metrics" className="card-surface p-6 group">
            <div className="text-xs uppercase tracking-[0.08em] text-primary font-medium mb-3">Metrics</div>
            <div className="font-medium text-lg mb-2">Training results</div>
            <div className="text-sm text-muted-foreground leading-relaxed">Reward 0.41 to 0.71 over 200 GRPO steps.</div>
          </Link>
          <Link href="/about" className="card-surface p-6 group">
            <div className="text-xs uppercase tracking-[0.08em] text-primary font-medium mb-3">About</div>
            <div className="font-medium text-lg mb-2">Team and architecture</div>
            <div className="text-sm text-muted-foreground leading-relaxed">3 bonus tracks, 4 citations, the full pipeline.</div>
          </Link>
        </div>
      </section>
    </div>
  );
}
