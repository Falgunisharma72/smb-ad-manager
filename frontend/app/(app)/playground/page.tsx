"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Play, Loader2, ArrowRight, Sparkles, AlertTriangle, CheckCircle2 } from "lucide-react";
import { FadeIn } from "@/components/fade-in";
import { apiReset, apiStep, type Observation, type StepResult, type TaskId } from "@/lib/api";
import { formatINR, formatInt } from "@/lib/utils";

const TASK_OPTIONS: { id: TaskId; label: string; sub: string }[] = [
  { id: "easy", label: "Easy", sub: "1-step reaction · no stress" },
  { id: "medium", label: "Medium", sub: "3 steps · 1 policy drift" },
  { id: "hard", label: "Hard", sub: "7 steps · 2 drifts + budget cut" },
];

const ACTION_PRESETS = [
  {
    key: "scale",
    label: "Scale winning campaign",
    desc: "Increase daily budget by 50%",
    builder: (obs: Observation) => {
      const c = obs.active_campaigns[0];
      if (!c) return null;
      return {
        tool: "update_budget",
        args: {
          campaign_id: c.id,
          new_daily_budget_inr: Math.round(c.daily_budget_inr * 1.5),
        },
        reasoning:
          "Metrics suggest strong ROAS — allocating more spend to this campaign.",
      };
    },
  },
  {
    key: "fetch",
    label: "Refresh metrics",
    desc: "Call get_metrics to get fresh data",
    builder: (obs: Observation) => {
      const c = obs.active_campaigns[0];
      if (!c) return null;
      return {
        tool: "get_metrics",
        args: { campaign_id: c.id },
        reasoning: "Fetching metrics before deciding — don't want to hallucinate.",
      };
    },
  },
  {
    key: "pause",
    label: "Pause campaign",
    desc: "Stop spending on the current campaign",
    builder: (obs: Observation) => {
      const c = obs.active_campaigns[0];
      if (!c) return null;
      return {
        tool: "pause_ad",
        args: { campaign_id: c.id },
        reasoning: "Preserving budget while reviewing strategy.",
      };
    },
  },
  {
    key: "noop",
    label: "Wait and observe",
    desc: "No action; let the campaign run another day",
    builder: () => ({
      tool: "noop",
      args: {},
      reasoning: "Observing current trends before committing.",
    }),
  },
];

export default function PlaygroundPage() {
  const [task, setTask] = useState<TaskId>("medium");
  const [obs, setObs] = useState<Observation | null>(null);
  const [history, setHistory] = useState<StepResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setLoading(true);
    setError(null);
    setHistory([]);
    try {
      const o = await apiReset(task, 42);
      setObs(o);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function takeAction(builder: (o: Observation) => { tool: string; args: Record<string, unknown>; reasoning: string } | null) {
    if (!obs) return;
    const action = builder(obs);
    if (!action) return;

    setLoading(true);
    setError(null);
    try {
      const result = await apiStep(action);
      setObs(result.observation);
      setHistory((h) => [...h, result]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const latestReward = history.length > 0 ? history[history.length - 1].reward : null;
  const firedHacks = (latestReward?.info?.fired_hacks || []) as string[];

  return (
    <div className="space-y-8">
      <FadeIn>
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl mb-2">
              <span className="text-gradient">Playground</span>
            </h1>
            <p className="text-muted-foreground">
              Pick a task tier, reset the environment, and watch the agent
              react to a live simulation of a Meta Ads campaign.
            </p>
          </div>
        </div>
      </FadeIn>

      {/* Task selector + run controls */}
      <FadeIn delay={0.1}>
        <div className="glass-card rounded-2xl p-6 space-y-5">
          <div className="grid grid-cols-3 gap-3">
            {TASK_OPTIONS.map((t) => (
              <motion.button
                key={t.id}
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setTask(t.id)}
                className={`text-left rounded-xl p-4 border transition ${
                  task === t.id
                    ? "border-primary bg-primary-soft/70 shadow-soft-md"
                    : "border-border bg-white/60 hover:border-primary/40"
                }`}
              >
                <div className="font-serif text-xl mb-1">{t.label}</div>
                <div className="text-xs text-muted-foreground">{t.sub}</div>
              </motion.button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={start}
              disabled={loading}
              className="btn btn-primary px-5 disabled:opacity-60"
            >
              {loading && !obs ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {obs ? "Restart episode" : "Start episode"}
            </button>
            {obs && (
              <span className="text-sm text-muted-foreground">
                Day <b className="text-foreground">{obs.day}</b> of{" "}
                <b className="text-foreground">{obs.days_total}</b> · step{" "}
                <b className="text-foreground">{obs.step}</b>
                {obs.step >= obs.days_total && (
                  <span className="ml-2 text-success font-medium">
                    · episode complete
                  </span>
                )}
              </span>
            )}
          </div>

          {error && (
            <div className="text-sm text-danger bg-danger/10 border border-danger/20 rounded-lg p-3">
              {error}
            </div>
          )}
        </div>
      </FadeIn>

      {/* Observation + actions */}
      <AnimatePresence mode="wait">
        {obs && (
          <motion.div
            key={obs.step}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35 }}
            className="grid lg:grid-cols-3 gap-6"
          >
            {/* SMB profile + campaign */}
            <div className="lg:col-span-2 space-y-5">
              <div className="rounded-2xl bg-white border border-border p-6 shadow-soft">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h3 className="font-serif text-2xl">{obs.smb_profile.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {obs.smb_profile.industry} · {obs.smb_profile.location}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">
                      Budget left
                    </div>
                    <div className="font-serif text-2xl text-primary">
                      {formatINR(obs.total_budget_remaining_inr)}
                    </div>
                  </div>
                </div>

                <p className="text-sm text-foreground/80 bg-muted/50 rounded-lg p-3 italic">
                  &ldquo;{obs.smb_profile.description}&rdquo;
                </p>

                {/* Active campaigns */}
                <div className="mt-5">
                  <div className="text-sm font-medium mb-2">Active campaigns</div>
                  <div className="space-y-2">
                    {obs.active_campaigns.length === 0 && (
                      <p className="text-sm text-muted-foreground">No campaigns.</p>
                    )}
                    {obs.active_campaigns.map((c) => {
                      const m = obs.latest_metrics[c.id];
                      const roas =
                        m && m.spend_inr > 0
                          ? (m.revenue_inr / m.spend_inr).toFixed(2)
                          : "—";
                      return (
                        <div
                          key={c.id}
                          className="rounded-lg border border-border bg-muted/30 p-3 flex items-center justify-between"
                        >
                          <div>
                            <div className="font-mono text-sm">{c.id}</div>
                            <div className="text-xs text-muted-foreground">
                              {c.objective} · daily {formatINR(c.daily_budget_inr)}
                            </div>
                          </div>
                          {m && (
                            <div className="text-right text-xs space-y-0.5">
                              <div>
                                <span className="text-muted-foreground">Imp: </span>
                                <b>{formatInt(m.impressions)}</b>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Rev: </span>
                                <b>{formatINR(m.revenue_inr)}</b>
                              </div>
                              <div>
                                <span className="text-muted-foreground">ROAS: </span>
                                <b
                                  className={
                                    Number(roas) >= 2
                                      ? "text-success"
                                      : Number(roas) < 1
                                      ? "text-danger"
                                      : ""
                                  }
                                >
                                  {roas}×
                                </b>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Recent events */}
                {obs.recent_events.length > 0 && (
                  <div className="mt-5">
                    <div className="text-sm font-medium mb-2">Recent events</div>
                    <ul className="space-y-1.5">
                      {obs.recent_events.map((e, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -6 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className={`text-sm flex items-start gap-2 ${
                            e.includes("⚠") || e.includes("FAILED")
                              ? "text-danger"
                              : e.includes("POLICY UPDATE")
                              ? "text-warning"
                              : "text-muted-foreground"
                          }`}
                        >
                          <span className="mt-1 w-1 h-1 rounded-full bg-current shrink-0" />
                          <span>{e}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Action menu + latest reward */}
            <div className="space-y-5">
              <div className="rounded-2xl bg-white border border-border p-6 shadow-soft">
                <h3 className="font-semibold mb-4">Choose an action</h3>
                <div className="space-y-2.5">
                  {ACTION_PRESETS.map((a) => {
                    const disabled =
                      loading ||
                      obs.step >= obs.days_total ||
                      !a.builder(obs);
                    return (
                      <motion.button
                        key={a.key}
                        whileHover={{ x: disabled ? 0 : 4 }}
                        whileTap={{ scale: 0.98 }}
                        disabled={disabled}
                        onClick={() => takeAction(a.builder)}
                        className={`w-full text-left rounded-xl border p-3 flex items-start gap-3 transition ${
                          disabled
                            ? "opacity-50 cursor-not-allowed border-border"
                            : "border-border hover:border-primary hover:bg-primary-soft/40"
                        }`}
                      >
                        <ArrowRight className="w-4 h-4 mt-0.5 text-primary shrink-0" />
                        <div>
                          <div className="font-medium text-sm">{a.label}</div>
                          <div className="text-xs text-muted-foreground">{a.desc}</div>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              {latestReward && (
                <motion.div
                  key={`reward-${history.length}`}
                  initial={{ scale: 0.98, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="rounded-2xl bg-white border border-border p-6 shadow-soft"
                >
                  <h3 className="font-semibold mb-3">Last reward</h3>
                  <div className="flex items-center gap-2 mb-4">
                    <div className="font-serif text-4xl text-primary">
                      {latestReward.score.toFixed(2)}
                    </div>
                    <div className="text-sm text-muted-foreground">/ 1.00</div>
                  </div>

                  <div className="space-y-1.5">
                    {Object.entries(latestReward.breakdown).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between text-xs"
                      >
                        <span className="text-muted-foreground">
                          {k.replace(/^r[0-9]_/, "").replaceAll("_", " ")}
                        </span>
                        <span
                          className={`font-mono tabular-nums ${
                            Number(v) >= 0.5 ? "text-success" : "text-danger"
                          }`}
                        >
                          {typeof v === "number"
                            ? v < 1 && v > 0
                              ? v.toFixed(2)
                              : String(v)
                            : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>

                  {firedHacks.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 rounded-lg bg-danger/10 border border-danger/20 p-3 flex items-start gap-2"
                    >
                      <AlertTriangle className="w-4 h-4 text-danger mt-0.5 shrink-0" />
                      <div className="text-xs">
                        <b className="text-danger">Reward hack caught:</b>{" "}
                        {firedHacks.join(", ")}
                      </div>
                    </motion.div>
                  )}

                  {firedHacks.length === 0 && latestReward.score >= 0.9 && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 rounded-lg bg-success/10 border border-success/20 p-3 flex items-start gap-2"
                    >
                      <CheckCircle2 className="w-4 h-4 text-success mt-0.5 shrink-0" />
                      <div className="text-xs text-success">
                        Clean action — all 5 reward components passed.
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </div>
          </motion.div>
        )}

        {!obs && !loading && !error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-dashed border-border p-10 text-center"
          >
            <Sparkles className="w-8 h-8 mx-auto mb-3 text-primary" />
            <p className="text-muted-foreground">
              Pick a task tier and click <b>Start episode</b> to begin.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
