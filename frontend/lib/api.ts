/**
 * Client for the HF Space backend.
 * All calls are read-mostly and don't need auth headers (CORS is open).
 */

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://falgunisharma-smb-ad-manager.hf.space";

export type TaskId = "easy" | "medium" | "hard";

export interface Observation {
  step: number;
  day: number;
  task_id: TaskId;
  smb_profile: {
    name: string;
    industry: string;
    location: string;
    monthly_budget_inr: number;
    goal: string;
    description: string;
  };
  total_budget_remaining_inr: number;
  days_elapsed: number;
  days_total: number;
  active_campaigns: Array<{
    id: string;
    objective: string;
    daily_budget_inr: number;
    status: string;
    created_day: number;
  }>;
  active_ads: Array<{
    id: string;
    ad_set_id: string;
    status: string;
    rejection_reason?: string | null;
    creative: {
      headline: string;
      body: string;
      image_description: string;
      call_to_action: string;
    };
  }>;
  recent_events: string[];
  latest_metrics: Record<
    string,
    {
      impressions: number;
      clicks: number;
      conversions: number;
      spend_inr: number;
      revenue_inr: number;
    }
  >;
  active_policies: Array<{
    id: string;
    name: string;
    description: string;
  }>;
}

export interface StepResult {
  observation: Observation;
  reward: {
    score: number;
    done: boolean;
    breakdown: {
      r1_roas_improvement: number;
      r2_policy_compliance: number;
      r3_format_compliance: number;
      r4_budget_discipline: number;
      r5_no_cheating: number;
    };
    info: {
      action_ok: boolean;
      action_error?: string | null;
      tool_result?: unknown;
      fired_hacks?: string[];
    };
  };
  done: boolean;
  info: Record<string, unknown>;
}

export async function apiReset(taskId: TaskId, seed: number | null = 42): Promise<Observation> {
  const res = await fetch(`${BASE_URL}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: taskId, seed }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
  return res.json();
}

export async function apiStep(action: {
  tool: string;
  args: Record<string, unknown>;
  reasoning: string;
}): Promise<StepResult> {
  const res = await fetch(`${BASE_URL}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Step failed: ${res.status}`);
  return res.json();
}

export async function apiHealth(): Promise<{ ok: boolean; service: string; version: string }> {
  const res = await fetch(`${BASE_URL}/healthz`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Healthz failed: ${res.status}`);
  return res.json();
}

export async function apiState(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/state`, { cache: "no-store" });
  if (!res.ok) throw new Error(`State failed: ${res.status}`);
  return res.json();
}
