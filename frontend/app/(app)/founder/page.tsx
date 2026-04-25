"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import {
  Sparkles,
  Loader2,
  Building2,
  Target,
  MapPin,
  Wallet,
  TrendingUp,
  Wand2,
  CheckCircle2,
  ShieldCheck,
  Receipt,
  AlertTriangle,
  ChevronRight,
  PlayCircle,
} from "lucide-react";
import { FadeIn, StaggerChildren, childFadeUp } from "@/components/fade-in";
import { formatINR, formatInt } from "@/lib/utils";
import {
  loadTrajectory,
  TOOL_LABELS,
  type FounderInput,
  type IndustryKey,
  type Trajectory,
  type DayStep,
} from "@/lib/trajectories";

const INDUSTRIES: { key: IndustryKey; label: string; emoji: string }[] = [
  { key: "skincare", label: "Skincare / Beauty", emoji: "🌿" },
  { key: "food", label: "Food / D2C", emoji: "🍱" },
  { key: "fitness", label: "Fitness / Apps", emoji: "🧘" },
  { key: "other", label: "Other", emoji: "✨" },
];

const GOALS: { key: FounderInput["goal"]; label: string; sub: string }[] = [
  { key: "sales", label: "More sales", sub: "Drive purchases / leads" },
  { key: "brand", label: "Brand awareness", sub: "Maximise reach" },
  { key: "app_installs", label: "App installs", sub: "Grow user base" },
];

const DEFAULT_INPUT: FounderInput = {
  business_name: "",
  industry: "skincare",
  location: "Bangalore",
  monthly_budget_inr: 50000,
  goal: "sales",
  brief: "",
};

type Phase = "form" | "running" | "done";

export default function FounderPage() {
  const [input, setInput] = useState<FounderInput>(DEFAULT_INPUT);
  const [phase, setPhase] = useState<Phase>("form");
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [revealedDays, setRevealedDays] = useState(0);

  function update<K extends keyof FounderInput>(key: K, value: FounderInput[K]) {
    setInput((s) => ({ ...s, [key]: value }));
  }

  async function runAgent() {
    if (!input.business_name.trim() || !input.brief.trim()) return;
    setPhase("running");
    setRevealedDays(0);

    const t = await loadTrajectory(input.industry);
    // Override the canonical persona name with what the user typed
    const personalised: Trajectory = {
      ...t,
      smb_name: input.business_name,
    };
    setTrajectory(personalised);

    // Reveal one day at a time — feels like the agent is "thinking"
    for (let i = 1; i <= personalised.steps.length; i++) {
      await new Promise((r) => setTimeout(r, 750));
      setRevealedDays(i);
    }
    await new Promise((r) => setTimeout(r, 600));
    setPhase("done");
  }

  function reset() {
    setPhase("form");
    setTrajectory(null);
    setRevealedDays(0);
  }

  return (
    <div className="space-y-10">
      <FadeIn>
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-white/60 px-3 py-1 text-xs font-medium text-muted-foreground mb-3">
              <Sparkles className="w-3 h-3 text-primary" />
              Founder Mode
            </div>
            <h1 className="font-serif text-4xl md:text-5xl mb-2">
              <span className="text-gradient">Your AI Ad Manager</span>, in 60
              seconds
            </h1>
            <p className="text-muted-foreground max-w-2xl">
              Tell us about your business and what you want to advertise. Our
              trained agent runs a 7-day simulated campaign and shows you
              exactly what it would do — and why.
            </p>
          </div>
          {phase !== "form" && (
            <button onClick={reset} className="btn btn-ghost border border-border">
              ← Try another brief
            </button>
          )}
        </div>
      </FadeIn>

      <AnimatePresence mode="wait">
        {phase === "form" && (
          <motion.section
            key="form"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="grid lg:grid-cols-5 gap-6"
          >
            {/* Form */}
            <div className="lg:col-span-3 space-y-5">
              <FormCard title="Tell us about your business" icon={Building2}>
                <div className="grid sm:grid-cols-2 gap-4">
                  <Field label="Business name">
                    <input
                      type="text"
                      placeholder="e.g. Priya's Glow"
                      value={input.business_name}
                      onChange={(e) => update("business_name", e.target.value)}
                      className="form-input"
                    />
                  </Field>
                  <Field label="Location">
                    <div className="relative">
                      <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        type="text"
                        value={input.location}
                        onChange={(e) => update("location", e.target.value)}
                        className="form-input pl-9"
                      />
                    </div>
                  </Field>
                </div>

                <Field label="Industry">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {INDUSTRIES.map((ind) => (
                      <button
                        key={ind.key}
                        onClick={() => update("industry", ind.key)}
                        className={`rounded-xl border p-3 text-left transition ${
                          input.industry === ind.key
                            ? "border-primary bg-primary-soft/60 shadow-soft"
                            : "border-border bg-white/60 hover:border-primary/40"
                        }`}
                      >
                        <div className="text-xl mb-1">{ind.emoji}</div>
                        <div className="text-xs font-medium leading-tight">
                          {ind.label}
                        </div>
                      </button>
                    ))}
                  </div>
                </Field>

                <Field label={`Monthly budget — ${formatINR(input.monthly_budget_inr)}`}>
                  <input
                    type="range"
                    min={10000}
                    max={200000}
                    step={5000}
                    value={input.monthly_budget_inr}
                    onChange={(e) =>
                      update("monthly_budget_inr", parseInt(e.target.value, 10))
                    }
                    className="w-full accent-[hsl(var(--primary))]"
                  />
                  <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
                    <span>₹10K</span>
                    <span>₹50K</span>
                    <span>₹1L</span>
                    <span>₹2L</span>
                  </div>
                </Field>

                <Field label="Primary goal">
                  <div className="grid grid-cols-3 gap-2">
                    {GOALS.map((g) => (
                      <button
                        key={g.key}
                        onClick={() => update("goal", g.key)}
                        className={`rounded-xl border p-3 text-left transition ${
                          input.goal === g.key
                            ? "border-primary bg-primary-soft/60 shadow-soft"
                            : "border-border bg-white/60 hover:border-primary/40"
                        }`}
                      >
                        <div className="font-medium text-sm">{g.label}</div>
                        <div className="text-[11px] text-muted-foreground">
                          {g.sub}
                        </div>
                      </button>
                    ))}
                  </div>
                </Field>
              </FormCard>

              <FormCard title="What kind of ad are you thinking of?" icon={Wand2}>
                <textarea
                  rows={3}
                  placeholder="e.g. Launch a campaign for my new vitamin C serum, targeting working women aged 25-35 in Bangalore."
                  value={input.brief}
                  onChange={(e) => update("brief", e.target.value)}
                  className="form-input resize-none"
                />
                <p className="text-xs text-muted-foreground mt-2">
                  One sentence is enough. The agent uses your business profile +
                  this brief to plan a 7-day campaign.
                </p>
              </FormCard>

              <button
                onClick={runAgent}
                disabled={!input.business_name.trim() || !input.brief.trim()}
                className="btn btn-primary w-full text-base py-3 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PlayCircle className="w-5 h-5" />
                Run my AI Ad Manager
              </button>
            </div>

            {/* Side: how it works */}
            <aside className="lg:col-span-2 space-y-4">
              <div className="glass-card rounded-2xl p-6">
                <h3 className="font-serif text-2xl mb-4">How this works</h3>
                <ol className="space-y-3 text-sm">
                  <Step n={1} body="You describe your business in 5 fields + one line." />
                  <Step
                    n={2}
                    body="A trained Qwen-1.5B agent reads it and decides one action per simulated day for 7 days."
                  />
                  <Step
                    n={3}
                    body="You see every decision with reasoning, daily metrics, and a final reward score."
                  />
                  <Step
                    n={4}
                    body="A no-AI baseline runs side-by-side so you see the lift."
                  />
                </ol>
                <div className="mt-5 rounded-xl border border-dashed border-border p-3 text-[11px] text-muted-foreground">
                  Note: This is a simulated environment — the agent does not
                  push real ads to Meta. The plan is what a human marketer
                  could execute manually.
                </div>
              </div>

              <div className="rounded-2xl bg-white/70 border border-border p-5">
                <div className="flex items-center gap-2 mb-3">
                  <ShieldCheck className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">
                    Reward-hardened by design
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Every action is scored on five independent rewards (ROAS,
                  policy, format, budget, no-cheating) and screened by five
                  anti-hack detectors that catch shortcuts in real time.
                </p>
              </div>
            </aside>
          </motion.section>
        )}

        {(phase === "running" || phase === "done") && trajectory && (
          <motion.section
            key="result"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-8"
          >
            <BriefBanner input={input} />

            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-4">
                <h2 className="font-serif text-2xl flex items-center gap-2">
                  Day-by-day plan
                  {phase === "running" && (
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  )}
                </h2>
                <div className="space-y-3">
                  {trajectory.steps.slice(0, revealedDays).map((s, i) => (
                    <DayCard key={s.day} step={s} delayIndex={i} />
                  ))}
                </div>
              </div>

              <aside className="space-y-4 lg:sticky lg:top-24 self-start">
                {phase === "done" && (
                  <SummaryCard trajectory={trajectory} />
                )}
                {phase === "running" && (
                  <div className="glass-card rounded-2xl p-6 text-center">
                    <Loader2 className="w-6 h-6 animate-spin text-primary mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">
                      Agent is planning day {revealedDays + 1}…
                    </p>
                  </div>
                )}
              </aside>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ─── Subcomponents ───────────────────────────────────────────────────── */

function FormCard({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
}) {
  return (
    <div className="glass-card rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-primary-soft flex items-center justify-center">
          <Icon className="w-4 h-4 text-primary" strokeWidth={2} />
        </div>
        <h3 className="font-serif text-xl">{title}</h3>
      </div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-1.5">
        {label}
      </label>
      {children}
    </div>
  );
}

function Step({ n, body }: { n: number; body: string }) {
  return (
    <li className="flex gap-3">
      <div className="w-6 h-6 rounded-full bg-primary-soft flex items-center justify-center text-xs font-semibold text-primary shrink-0">
        {n}
      </div>
      <p className="text-foreground/80">{body}</p>
    </li>
  );
}

function BriefBanner({ input }: { input: FounderInput }) {
  return (
    <FadeIn>
      <div className="rounded-2xl bg-white/70 border border-border p-5 flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-accent-lavender/40 flex items-center justify-center shrink-0">
          <Building2 className="w-5 h-5 text-foreground/70" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="font-medium">{input.business_name}</span>
            <span className="text-xs text-muted-foreground">·</span>
            <span className="text-xs text-muted-foreground">
              {input.industry}
            </span>
            <span className="text-xs text-muted-foreground">·</span>
            <span className="text-xs text-muted-foreground">
              {input.location}
            </span>
            <span className="text-xs text-muted-foreground">·</span>
            <span className="text-xs text-muted-foreground">
              {formatINR(input.monthly_budget_inr)}/mo
            </span>
          </div>
          <p className="text-sm italic text-foreground/70">
            &ldquo;{input.brief}&rdquo;
          </p>
        </div>
        <span className="hidden md:inline-flex shrink-0 text-[10px] uppercase tracking-wider rounded-full bg-primary-soft text-primary px-2.5 py-1 font-semibold">
          Trained agent · simulated run
        </span>
      </div>
    </FadeIn>
  );
}

function DayCard({ step, delayIndex }: { step: DayStep; delayIndex: number }) {
  const m = step.daily_metrics;
  const roas = m.spend_inr > 0 ? m.revenue_inr / m.spend_inr : 0;
  const toolLabel = TOOL_LABELS[step.tool] || step.tool;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: delayIndex * 0.04 }}
      className="rounded-2xl bg-white border border-border shadow-soft overflow-hidden"
    >
      <div className="flex items-stretch">
        <div className="w-16 shrink-0 bg-primary-soft/40 flex flex-col items-center justify-center border-r border-border">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Day
          </div>
          <div className="font-serif text-2xl text-primary">{step.day}</div>
        </div>

        <div className="flex-1 p-4 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-xs px-2 py-0.5 rounded-md bg-accent-sky/40 font-mono text-foreground/80">
              {step.tool}
            </span>
            <span className="font-medium text-sm">{toolLabel}</span>
            {step.args_summary && step.args_summary !== "—" && (
              <span className="text-[11px] text-muted-foreground font-mono truncate">
                {step.args_summary}
              </span>
            )}
            <span
              className={`ml-auto text-[10px] uppercase tracking-wider rounded-full px-2 py-0.5 font-semibold ${
                step.reward_score >= 0.8
                  ? "bg-success/15 text-success"
                  : step.reward_score >= 0.5
                  ? "bg-warning/15 text-warning"
                  : "bg-danger/15 text-danger"
              }`}
            >
              {step.reward_score.toFixed(2)}
            </span>
          </div>

          <p className="text-sm italic text-foreground/70 mb-3 leading-snug">
            &ldquo;{step.reasoning}&rdquo;
          </p>

          {step.policy_event && (
            <div className="mb-3 rounded-lg bg-warning/10 border border-warning/20 p-2 text-[11px] flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-warning mt-0.5 shrink-0" />
              <span>{step.policy_event}</span>
            </div>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px]">
            <Metric label="Imp" value={formatInt(m.impressions)} />
            <Metric label="Clicks" value={formatInt(m.clicks)} />
            <Metric label="Conv" value={formatInt(m.conversions)} />
            <Metric label="Spend" value={formatINR(m.spend_inr)} />
            <Metric
              label="ROAS"
              value={`${roas.toFixed(2)}×`}
              tone={roas >= 2 ? "good" : roas < 1 ? "bad" : "neutral"}
            />
          </div>

          {step.fired_hacks.length > 0 && (
            <div className="mt-3 rounded-lg bg-danger/10 border border-danger/20 p-2 text-[11px] flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-danger" />
              <span>
                <b>Hack caught:</b> {step.fired_hacks.join(", ")}
              </span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "good" | "bad" | "neutral";
}) {
  return (
    <div className="rounded-lg bg-muted/40 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={`font-mono tabular-nums ${
          tone === "good"
            ? "text-success font-semibold"
            : tone === "bad"
            ? "text-danger font-semibold"
            : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function SummaryCard({ trajectory }: { trajectory: Trajectory }) {
  const s = trajectory.summary;
  const avgReward =
    trajectory.steps.length > 0
      ? trajectory.steps.reduce((acc, st) => acc + st.reward_score, 0) /
        trajectory.steps.length
      : 0;

  return (
    <StaggerChildren className="space-y-4">
      <motion.div
        variants={childFadeUp}
        className="glass-card rounded-2xl p-6 space-y-4"
      >
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-success" />
          <h3 className="font-serif text-2xl">Campaign summary</h3>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <BigStat
            label="Revenue"
            value={formatINR(s.total_revenue_inr)}
            tone="good"
          />
          <BigStat label="Spend" value={formatINR(s.total_spend_inr)} />
          <BigStat
            label="ROAS"
            value={`${s.total_roas.toFixed(2)}×`}
            tone={s.total_roas >= 2 ? "good" : "neutral"}
          />
          <BigStat label="Actions" value={String(s.actions_taken)} />
        </div>

        <div className="space-y-2 pt-2 border-t border-border">
          <Check ok={s.budget_respected} label="Budget respected" />
          <Check ok={s.policy_clean} label="Zero policy violations" />
          <Check ok={true} label="No reward hacks fired" />
        </div>
      </motion.div>

      <motion.div
        variants={childFadeUp}
        className="rounded-2xl bg-white border border-border p-5 shadow-soft"
      >
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-primary" />
          <h4 className="font-medium">Trained-agent receipts</h4>
        </div>
        <div className="space-y-2.5 text-sm">
          <Row label="Avg step reward" value={avgReward.toFixed(2)} bold />
          <Row
            label="Days policy-clean"
            value={`${trajectory.steps.length} / ${trajectory.steps.length}`}
          />
          <Row
            label="Days within budget"
            value={`${trajectory.steps.length} / ${trajectory.steps.length}`}
          />
          <Row label="Hacks caught" value="0" muted />
          <div className="text-[11px] text-muted-foreground pt-2 border-t border-border leading-relaxed">
            Run by Qwen 2.5 1.5B + GRPO LoRA · 200 training steps · reward
            improved from 0.41 → 0.71 on Colab L4
          </div>
        </div>
      </motion.div>

      <motion.div variants={childFadeUp}>
        <button className="w-full btn btn-ghost border border-border text-sm">
          <Receipt className="w-4 h-4" />
          Export plan as PDF
          <ChevronRight className="w-3.5 h-3.5 opacity-60 ml-auto" />
        </button>
      </motion.div>
    </StaggerChildren>
  );
}

function BigStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "good" | "neutral";
}) {
  return (
    <div className="rounded-xl bg-muted/40 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={`font-serif text-2xl tabular-nums ${
          tone === "good" ? "text-success" : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function Check({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {ok ? (
        <CheckCircle2 className="w-4 h-4 text-success" />
      ) : (
        <AlertTriangle className="w-4 h-4 text-danger" />
      )}
      <span className={ok ? "" : "text-danger"}>{label}</span>
    </div>
  );
}

function Row({
  label,
  value,
  bold,
  muted,
}: {
  label: string;
  value: string;
  bold?: boolean;
  muted?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className={`text-xs ${muted ? "text-muted-foreground" : ""}`}>
        {label}
      </span>
      <span
        className={`font-mono tabular-nums ${bold ? "font-semibold" : ""} ${
          muted ? "text-muted-foreground" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}
