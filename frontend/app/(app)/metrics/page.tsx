"use client";

import { motion } from "framer-motion";
import {
  TrendingUp,
  Activity,
  Clock,
  Cpu,
  Zap,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  BookOpen,
  Sparkles,
  GitBranch,
  Target,
  ImageIcon,
} from "lucide-react";
import { FadeIn, StaggerChildren, childFadeUp } from "@/components/fade-in";

const TRAINING_FACTS = [
  { icon: TrendingUp, label: "Reward improvement", value: "+73%", sub: "0.41 → 0.71" },
  { icon: Clock, label: "Total runtime", value: "51 min", sub: "200 steps" },
  { icon: Cpu, label: "Hardware", value: "1× L4", sub: "Colab Pro · 22 GB" },
  { icon: Zap, label: "Algorithm", value: "2-GRPO", sub: "group size 2" },
];

const CHARTS = [
  {
    file: "1.png",
    title: "Reward / mean",
    caption:
      "Reward climbs from 0.41 to 0.71 over 200 steps. The clean monotonic curve is the textbook RL convergence signature — no collapse, no plateau.",
    tag: "headline",
  },
  {
    file: "2.png",
    title: "Policy loss",
    caption:
      "Policy loss oscillates near zero — expected for GRPO, where the loss reflects advantage-weighted log-probs of sampled completions.",
  },
  {
    file: "3.png",
    title: "Entropy",
    caption:
      "Entropy increased from 0.29 to 0.47 over training — the model didn't collapse to deterministic outputs. Healthy exploration was preserved.",
  },
  {
    file: "4.png",
    title: "Completion length",
    caption:
      "Completions stayed in the 90-120 token band — concise, well-formed JSON actions. No mode collapse to single-token spam, no length explosion.",
  },
];

const REWARD_AVERAGES = [
  { id: "r1", name: "ROAS improvement", avg: 0.62, color: "primary" },
  { id: "r2", name: "Policy compliance", avg: 1.0, color: "success" },
  { id: "r3", name: "Format compliance", avg: 0.0, color: "danger", note: "honest weak spot" },
  { id: "r4", name: "Budget discipline", avg: 1.0, color: "success" },
  { id: "r5", name: "No cheating", avg: 1.0, color: "success" },
];

const STAGE_COMPARISON = [
  {
    stage: "Base Qwen 2.5 1.5B",
    parses_json: "—",
    follows_schema: "—",
    policy_clean: "—",
    avg_reward: "n/a",
    note: "Untrained on this task — outputs unstructured prose.",
  },
  {
    stage: "+ SFT warm-start",
    parses_json: "✓",
    follows_schema: "partial",
    policy_clean: "✓",
    avg_reward: "0.41",
    note: "Learned the JSON action format from 100 hand-crafted examples (3 epochs).",
  },
  {
    stage: "+ GRPO (final)",
    parses_json: "✓",
    follows_schema: "partial",
    policy_clean: "✓",
    avg_reward: "0.71",
    note: "RL refinement — better reasoning text, more confident tool selection, perfect policy compliance.",
  },
];

const LIMITATIONS = [
  {
    finding: "Agent converges to a 'spam create' policy",
    detail:
      "After GRPO the model heavily prefers create_campaign / create_ad. These score reliable partial credit (tool present + args present + reasoning present) but rarely modify the running campaign — so lift over a noop baseline is small.",
    fix: "Train longer (1k+ steps) or remove partial credit so the agent can't score without genuine campaign improvement.",
  },
  {
    finding: "Format compliance reward (r3) stays at 0.0",
    detail:
      "The model outputs valid JSON but uses field names like daily_budget instead of daily_budget_inr. The strict Pydantic schema rejects these as invalid args even though the JSON parses.",
    fix: "Either soften r3 to 'parses as dict + has expected top-level keys', or include schema-correct examples in the SFT data.",
  },
  {
    finding: "Hallucinated tool names",
    detail:
      "Some seeds produce tool names like creative_curation, creative_selection — invented by the model, rejected by the env with 422.",
    fix: "Include the explicit tool list in every prompt (we currently rely on the model remembering it from SFT).",
  },
];

const CITATIONS = [
  {
    title: "DeepSeek-R1 (GRPO origin)",
    venue: "arxiv 2501.12948",
    url: "https://arxiv.org/abs/2501.12948",
  },
  {
    title: "2-GRPO: 12.5% rollouts of standard GRPO",
    venue: "arxiv 2510.00977",
    url: "https://arxiv.org/abs/2510.00977",
  },
  {
    title: "OpenEnv specification",
    venue: "huggingface/openenv",
    url: "https://github.com/huggingface/openenv",
  },
];

const WANDB_RUN_URL =
  "https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc";

export default function MetricsPage() {
  return (
    <div className="space-y-12">
      {/* Hero */}
      <FadeIn>
        <div className="relative overflow-hidden rounded-2xl glass-card p-8 md:p-12">
          <div className="relative z-10 max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-white/60 px-3 py-1 text-xs font-medium text-muted-foreground mb-4">
              <Activity className="w-3 h-3 text-primary" />
              Training run · volcanic-blaze-7
            </div>
            <h1 className="font-serif text-4xl md:text-5xl mb-3">
              <span className="text-gradient">200 GRPO steps</span>, 51 minutes,
              one L4.
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed mb-6">
              Reward climbed from <b className="text-foreground">0.41</b> to{" "}
              <b className="text-foreground">0.71</b> — a 73% improvement on a
              5-component reward signal that already includes 5 anti-hack
              detectors.
            </p>
            <a
              href={WANDB_RUN_URL}
              target="_blank"
              rel="noreferrer"
              className="btn btn-primary"
            >
              <Activity className="w-4 h-4" />
              Open the W&amp;B run
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
          <motion.div
            aria-hidden
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -top-12 -right-10 w-44 h-44 rounded-full bg-primary-soft/60 blur-3xl"
          />
        </div>
      </FadeIn>

      {/* Headline facts */}
      <section>
        <StaggerChildren className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {TRAINING_FACTS.map((f) => {
            const Icon = f.icon;
            return (
              <motion.div
                key={f.label}
                variants={childFadeUp}
                className="rounded-2xl bg-white/80 border border-border p-5 shadow-soft"
              >
                <Icon className="w-5 h-5 text-primary mb-3" strokeWidth={1.8} />
                <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">
                  {f.label}
                </div>
                <div className="font-serif text-3xl text-foreground">
                  {f.value}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {f.sub}
                </div>
              </motion.div>
            );
          })}
        </StaggerChildren>
      </section>

      {/* Charts */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-2">Training curves</h2>
          <p className="text-muted-foreground mb-5">
            Direct screenshots from the W&amp;B run. No smoothing tricks.
          </p>
        </FadeIn>
        <StaggerChildren className="grid md:grid-cols-2 gap-5">
          {CHARTS.map((c) => (
            <motion.div
              key={c.file}
              variants={childFadeUp}
              className={`rounded-2xl bg-white border ${
                c.tag === "headline" ? "border-primary/40 shadow-soft-md" : "border-border shadow-soft"
              } overflow-hidden`}
            >
              <ChartImage file={c.file} title={c.title} />
              <div className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="font-semibold">{c.title}</h3>
                  {c.tag === "headline" && (
                    <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-primary-soft text-primary font-semibold">
                      headline
                    </span>
                  )}
                </div>
                <p className="text-sm text-foreground/70 leading-relaxed">
                  {c.caption}
                </p>
              </div>
            </motion.div>
          ))}
        </StaggerChildren>
      </section>

      {/* Per-component reward breakdown */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-2">
            5-component reward breakdown
          </h2>
          <p className="text-muted-foreground mb-5">
            Average score per component over the trained agent's runs against
            the live env.
          </p>
        </FadeIn>
        <div className="rounded-2xl bg-white border border-border shadow-soft p-2">
          {REWARD_AVERAGES.map((r) => (
            <RewardBar key={r.id} {...r} />
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          r2/r4/r5 saturate at 1.0 — the agent is provably policy-clean,
          budget-clean, and doesn't fabricate metrics. r1 (ROAS) is the
          variable. r3 (format) is the honest weakness — see Limitations.
        </p>
      </section>

      {/* Stage comparison table */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-2">SFT → GRPO progression</h2>
          <p className="text-muted-foreground mb-5">
            Same model, three checkpoints. Each row is a stage in the training
            pipeline.
          </p>
        </FadeIn>
        <div className="rounded-2xl bg-white border border-border shadow-soft overflow-hidden">
          <div className="hidden md:grid grid-cols-12 gap-3 px-5 py-3 bg-muted/30 border-b border-border text-xs font-medium uppercase tracking-wider text-muted-foreground">
            <div className="col-span-3">Stage</div>
            <div className="col-span-2">Parses JSON</div>
            <div className="col-span-2">Schema</div>
            <div className="col-span-2">Policy clean</div>
            <div className="col-span-1">Reward</div>
            <div className="col-span-2">Notes</div>
          </div>
          {STAGE_COMPARISON.map((row, i) => (
            <div
              key={row.stage}
              className={`grid grid-cols-1 md:grid-cols-12 gap-3 px-5 py-4 ${
                i < STAGE_COMPARISON.length - 1 ? "border-b border-border" : ""
              } ${i === 2 ? "bg-primary-soft/15" : ""}`}
            >
              <div className="md:col-span-3 font-medium text-sm flex items-center gap-2">
                {i === 2 && <Sparkles className="w-3.5 h-3.5 text-primary" />}
                {row.stage}
              </div>
              <div className="md:col-span-2 text-sm">
                <Badge value={row.parses_json} />
              </div>
              <div className="md:col-span-2 text-sm">
                <Badge value={row.follows_schema} />
              </div>
              <div className="md:col-span-2 text-sm">
                <Badge value={row.policy_clean} />
              </div>
              <div className="md:col-span-1 text-sm font-mono tabular-nums font-semibold">
                {row.avg_reward}
              </div>
              <div className="md:col-span-2 text-xs text-muted-foreground leading-snug">
                {row.note}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Honest limitations */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-2 flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-warning" />
            Limitations & future work
          </h2>
          <p className="text-muted-foreground mb-5">
            Honest read of what the model didn't learn — and what we'd fix with
            more compute.
          </p>
        </FadeIn>
        <StaggerChildren className="space-y-3">
          {LIMITATIONS.map((l) => (
            <motion.article
              key={l.finding}
              variants={childFadeUp}
              className="rounded-2xl bg-white border border-border p-5 shadow-soft"
            >
              <h3 className="font-medium text-base mb-2 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-warning" />
                {l.finding}
              </h3>
              <p className="text-sm text-foreground/75 mb-2 leading-relaxed">
                {l.detail}
              </p>
              <p className="text-xs text-primary font-medium flex items-start gap-1.5">
                <Target className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>
                  <b>Fix:</b> {l.fix}
                </span>
              </p>
            </motion.article>
          ))}
        </StaggerChildren>
      </section>

      {/* Citations */}
      <section>
        <FadeIn delay={0.05}>
          <h2 className="font-serif text-3xl mb-5 flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-primary" />
            Method & citations
          </h2>
        </FadeIn>
        <div className="rounded-2xl bg-white border border-border divide-y divide-border overflow-hidden shadow-soft">
          {CITATIONS.map((c) => (
            <a
              key={c.title}
              href={c.url}
              target="_blank"
              rel="noreferrer"
              className="p-4 flex items-center gap-4 hover:bg-muted/20 transition group"
            >
              <div className="w-9 h-9 rounded-lg bg-primary-soft/60 flex items-center justify-center shrink-0">
                <GitBranch className="w-4 h-4 text-primary" />
              </div>
              <div className="flex-1">
                <div className="font-medium group-hover:text-primary transition">
                  {c.title}
                </div>
                <div className="text-xs text-muted-foreground font-mono">
                  {c.venue}
                </div>
              </div>
              <ExternalLink className="w-4 h-4 text-muted-foreground" />
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}

/* ─── Subcomponents ────────────────────────────────────────────────── */

function ChartImage({ file, title }: { file: string; title: string }) {
  // Use a plain <img> so missing files render gracefully via onError swap.
  return (
    <div className="aspect-[16/10] bg-gradient-to-br from-muted/40 to-muted/10 relative">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`/charts/${file}`}
        alt={title}
        className="absolute inset-0 w-full h-full object-contain p-3"
        onError={(e) => {
          const img = e.currentTarget;
          img.style.display = "none";
          const fallback = img.nextElementSibling as HTMLElement | null;
          if (fallback) fallback.style.display = "flex";
        }}
      />
      <div
        className="absolute inset-0 hidden flex-col items-center justify-center text-center px-6 text-muted-foreground"
        style={{ display: "none" }}
      >
        <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
        <div className="text-sm font-medium mb-1">{title}</div>
        <div className="text-xs">
          Drop <code className="px-1 py-0.5 bg-muted rounded">{file}</code> in{" "}
          <code className="px-1 py-0.5 bg-muted rounded">
            /public/charts/
          </code>
        </div>
      </div>
    </div>
  );
}

function RewardBar({
  id,
  name,
  avg,
  color,
  note,
}: {
  id: string;
  name: string;
  avg: number;
  color: string;
  note?: string;
}) {
  const pct = Math.round(avg * 100);
  const colorClass =
    color === "success"
      ? "bg-success"
      : color === "danger"
      ? "bg-danger"
      : "bg-primary";
  return (
    <div className="px-4 py-3 grid grid-cols-12 items-center gap-3">
      <div className="col-span-3 md:col-span-2">
        <div className="font-mono text-xs text-muted-foreground">{id}</div>
        <div className="font-medium text-sm">{name}</div>
      </div>
      <div className="col-span-7 md:col-span-8">
        <div className="h-2.5 rounded-full bg-muted overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className={`h-full rounded-full ${colorClass}`}
          />
        </div>
        {note && (
          <div className="text-[11px] text-muted-foreground mt-1">{note}</div>
        )}
      </div>
      <div className="col-span-2 text-right font-mono tabular-nums font-semibold">
        {avg.toFixed(2)}
      </div>
    </div>
  );
}

function Badge({ value }: { value: string }) {
  if (value === "✓")
    return (
      <span className="inline-flex items-center gap-1 text-success font-medium">
        <CheckCircle2 className="w-3.5 h-3.5" /> yes
      </span>
    );
  if (value === "—")
    return <span className="text-muted-foreground">—</span>;
  return <span>{value}</span>;
}
