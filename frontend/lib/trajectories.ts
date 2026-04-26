/**
 * Trajectory data - pre-recorded runs of the trained agent on representative
 * SMB personas. Used by /founder page to "replay" realistic agent behaviour.
 *
 * After GRPO training finishes, replace these mocks with real model outputs
 * (run inference offline → save same shape as JSON files).
 */

export type IndustryKey = "skincare" | "food" | "fitness" | "other";

export interface FounderInput {
  business_name: string;
  industry: IndustryKey;
  location: string;
  monthly_budget_inr: number;
  goal: "sales" | "brand" | "app_installs";
  brief: string;
}

export interface DayStep {
  day: number;
  tool: string;
  args_summary: string;
  reasoning: string;
  daily_metrics: {
    impressions: number;
    clicks: number;
    conversions: number;
    spend_inr: number;
    revenue_inr: number;
  };
  reward_score: number;
  reward_breakdown: {
    r1_roas: number;
    r2_policy: number;
    r3_format: number;
    r4_budget: number;
    r5_no_cheating: number;
  };
  fired_hacks: string[];
  policy_event?: string;
}

export interface Trajectory {
  persona: IndustryKey;
  smb_name: string;
  steps: DayStep[];
  summary: {
    total_spend_inr: number;
    total_revenue_inr: number;
    total_roas: number;
    budget_respected: boolean;
    policy_clean: boolean;
    actions_taken: number;
  };
  baseline: {
    /** What the same SMB would get with NO optimization (noop everything). */
    total_spend_inr: number;
    total_revenue_inr: number;
    total_roas: number;
  };
}

const SKINCARE: Trajectory = {
  persona: "skincare",
  smb_name: "Priya's Glow",
  steps: [
    {
      day: 1,
      tool: "create_campaign",
      args_summary: "objective=conversions, daily=₹1,667",
      reasoning:
        "Starting with a conservative daily budget - 1/30th of monthly to leave room for scaling winners.",
      daily_metrics: { impressions: 4200, clicks: 78, conversions: 6, spend_inr: 1620, revenue_inr: 2460 },
      reward_score: 0.61,
      reward_breakdown: { r1_roas: 0.45, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 2,
      tool: "noop",
      args_summary: "-",
      reasoning:
        "Day 1 ROAS = 1.52×; promising but noisy. Letting it run another day before committing more budget.",
      daily_metrics: { impressions: 5180, clicks: 95, conversions: 9, spend_inr: 1655, revenue_inr: 3690 },
      reward_score: 0.71,
      reward_breakdown: { r1_roas: 0.62, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 3,
      tool: "get_metrics",
      args_summary: "campaign_id=c001",
      reasoning: "Fetching fresh metrics before scaling - never act on stale data.",
      daily_metrics: { impressions: 5320, clicks: 102, conversions: 11, spend_inr: 1655, revenue_inr: 4510 },
      reward_score: 0.83,
      reward_breakdown: { r1_roas: 0.78, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 4,
      tool: "update_budget",
      args_summary: "campaign_id=c001, new_daily=₹2,500",
      reasoning:
        "Three-day ROAS holding at 2.7× - scaling up 50%. Within monthly budget envelope.",
      daily_metrics: { impressions: 7800, clicks: 148, conversions: 16, spend_inr: 2480, revenue_inr: 6720 },
      reward_score: 0.89,
      reward_breakdown: { r1_roas: 0.84, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 5,
      tool: "get_policy_updates",
      args_summary: "-",
      reasoning:
        "Policy drift detected - health-adjacent ads now require a disclaimer. Need to read the full rule before reacting.",
      daily_metrics: { impressions: 7950, clicks: 144, conversions: 14, spend_inr: 2495, revenue_inr: 5880 },
      reward_score: 0.79,
      reward_breakdown: { r1_roas: 0.71, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
      policy_event: "POLICY UPDATE: Health-adjacent ads must include '*not medical advice'",
    },
    {
      day: 6,
      tool: "rewrite_creative",
      args_summary: "ad_id=ad001, added disclaimer to body",
      reasoning:
        "Adding required disclaimer to comply with new policy. Headline kept identical to preserve learnings.",
      daily_metrics: { impressions: 7600, clicks: 138, conversions: 15, spend_inr: 2475, revenue_inr: 6300 },
      reward_score: 0.91,
      reward_breakdown: { r1_roas: 0.82, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 7,
      tool: "noop",
      args_summary: "-",
      reasoning:
        "Performance recovered post-rewrite. Letting the creative bake; will report final metrics tomorrow.",
      daily_metrics: { impressions: 7820, clicks: 142, conversions: 17, spend_inr: 2490, revenue_inr: 7140 },
      reward_score: 0.94,
      reward_breakdown: { r1_roas: 0.88, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
  ],
  summary: {
    total_spend_inr: 14870,
    total_revenue_inr: 36700,
    total_roas: 2.47,
    budget_respected: true,
    policy_clean: true,
    actions_taken: 5,
  },
  baseline: { total_spend_inr: 14870, total_revenue_inr: 21300, total_roas: 1.43 },
};

const FOOD: Trajectory = {
  persona: "food",
  smb_name: "Amma's Kitchen",
  steps: [
    {
      day: 1,
      tool: "create_campaign",
      args_summary: "objective=conversions, daily=₹1,000",
      reasoning:
        "Food category - UGC-heavy creative usually wins. Starting modest to gauge initial CTR.",
      daily_metrics: { impressions: 3100, clicks: 62, conversions: 7, spend_inr: 980, revenue_inr: 2310 },
      reward_score: 0.74,
      reward_breakdown: { r1_roas: 0.66, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.96, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 2,
      tool: "get_metrics",
      args_summary: "campaign_id=c001",
      reasoning: "Strong ROAS (2.36×) - verifying numbers before scaling up.",
      daily_metrics: { impressions: 3420, clicks: 70, conversions: 8, spend_inr: 985, revenue_inr: 2640 },
      reward_score: 0.81,
      reward_breakdown: { r1_roas: 0.74, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.96, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 3,
      tool: "update_budget",
      args_summary: "campaign_id=c001, new_daily=₹1,800",
      reasoning: "Two-day ROAS confirmed - scaling 80%. Conservative given budget envelope.",
      daily_metrics: { impressions: 6100, clicks: 122, conversions: 14, spend_inr: 1780, revenue_inr: 4620 },
      reward_score: 0.87,
      reward_breakdown: { r1_roas: 0.81, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.93, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 4,
      tool: "noop",
      args_summary: "-",
      reasoning: "ROAS holding at 2.6× post-scale. Letting the algorithm learn before another bump.",
      daily_metrics: { impressions: 6280, clicks: 119, conversions: 15, spend_inr: 1795, revenue_inr: 4950 },
      reward_score: 0.88,
      reward_breakdown: { r1_roas: 0.83, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.93, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 5,
      tool: "get_metrics",
      args_summary: "campaign_id=c001",
      reasoning: "Mid-week check - looking for signs of fatigue before final push.",
      daily_metrics: { impressions: 6320, clicks: 116, conversions: 14, spend_inr: 1790, revenue_inr: 4620 },
      reward_score: 0.85,
      reward_breakdown: { r1_roas: 0.78, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.93, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 6,
      tool: "noop",
      args_summary: "-",
      reasoning: "Slight CTR dip but conversions stable. Not pulling the trigger on rewrite yet.",
      daily_metrics: { impressions: 6190, clicks: 113, conversions: 15, spend_inr: 1780, revenue_inr: 4920 },
      reward_score: 0.87,
      reward_breakdown: { r1_roas: 0.81, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.93, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 7,
      tool: "noop",
      args_summary: "-",
      reasoning: "Stable end-of-week. Final ROAS lands at 2.7× - strong week for a first campaign.",
      daily_metrics: { impressions: 6350, clicks: 121, conversions: 16, spend_inr: 1795, revenue_inr: 5280 },
      reward_score: 0.91,
      reward_breakdown: { r1_roas: 0.86, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.93, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
  ],
  summary: {
    total_spend_inr: 10905,
    total_revenue_inr: 29340,
    total_roas: 2.69,
    budget_respected: true,
    policy_clean: true,
    actions_taken: 3,
  },
  baseline: { total_spend_inr: 10905, total_revenue_inr: 16400, total_roas: 1.50 },
};

const FITNESS: Trajectory = {
  persona: "fitness",
  smb_name: "PowerYoga",
  steps: [
    {
      day: 1,
      tool: "create_campaign",
      args_summary: "objective=app_installs, daily=₹2,667",
      reasoning: "App-installs objective fits the goal. Starting at 1/30th of monthly cap.",
      daily_metrics: { impressions: 18200, clicks: 312, conversions: 24, spend_inr: 2640, revenue_inr: 1980 },
      reward_score: 0.42,
      reward_breakdown: { r1_roas: 0.32, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 2,
      tool: "get_metrics",
      args_summary: "campaign_id=c001",
      reasoning: "ROAS 0.75× - concerning. Pulling fresh data to see if it's signal or noise.",
      daily_metrics: { impressions: 18900, clicks: 305, conversions: 22, spend_inr: 2650, revenue_inr: 1815 },
      reward_score: 0.39,
      reward_breakdown: { r1_roas: 0.28, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 3,
      tool: "rewrite_creative",
      args_summary: "ad_id=ad001, new headline emphasises beginner-friendly framing",
      reasoning:
        "Two-day ROAS under 1× - creative not resonating. Rewriting toward beginner audience instead of advanced yogis.",
      daily_metrics: { impressions: 17400, clicks: 380, conversions: 31, spend_inr: 2620, revenue_inr: 2790 },
      reward_score: 0.61,
      reward_breakdown: { r1_roas: 0.51, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 4,
      tool: "noop",
      args_summary: "-",
      reasoning: "Click rate jumped 25% post-rewrite. Letting it run one more day to confirm.",
      daily_metrics: { impressions: 17900, clicks: 410, conversions: 38, spend_inr: 2660, revenue_inr: 3420 },
      reward_score: 0.74,
      reward_breakdown: { r1_roas: 0.65, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.95, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 5,
      tool: "update_budget",
      args_summary: "campaign_id=c001, new_daily=₹4,000",
      reasoning:
        "Rewrite worked - ROAS climbed to 1.29×. Scaling 50% to capture momentum while creative is fresh.",
      daily_metrics: { impressions: 26200, clicks: 612, conversions: 58, spend_inr: 3980, revenue_inr: 5220 },
      reward_score: 0.78,
      reward_breakdown: { r1_roas: 0.71, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 6,
      tool: "noop",
      args_summary: "-",
      reasoning: "ROAS holding at 1.31×. Conversions stable. Letting the algorithm bake.",
      daily_metrics: { impressions: 26800, clicks: 624, conversions: 61, spend_inr: 3990, revenue_inr: 5490 },
      reward_score: 0.81,
      reward_breakdown: { r1_roas: 0.74, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
    {
      day: 7,
      tool: "noop",
      args_summary: "-",
      reasoning: "Closing the week clean. Net positive turnaround from a slow start.",
      daily_metrics: { impressions: 27100, clicks: 628, conversions: 62, spend_inr: 3995, revenue_inr: 5580 },
      reward_score: 0.83,
      reward_breakdown: { r1_roas: 0.76, r2_policy: 1.0, r3_format: 1.0, r4_budget: 0.92, r5_no_cheating: 1.0 },
      fired_hacks: [],
    },
  ],
  summary: {
    total_spend_inr: 22535,
    total_revenue_inr: 26295,
    total_roas: 1.17,
    budget_respected: true,
    policy_clean: true,
    actions_taken: 4,
  },
  baseline: { total_spend_inr: 22535, total_revenue_inr: 12400, total_roas: 0.55 },
};

export const TRAJECTORIES: Record<IndustryKey, Trajectory> = {
  skincare: SKINCARE,
  food: FOOD,
  fitness: FITNESS,
  other: SKINCARE, // fallback to skincare for "other" until we have a generic one
};

/**
 * Fetch a real trained-model trajectory from /public/trajectories/{key}.json.
 * Falls back to the inline mock if the file isn't there.
 *
 * The JSON files in public/trajectories/ are real outputs from the trained
 * Qwen 1.5B + GRPO LoRA, replayed in the env. Generated offline via the
 * Cell G inference loop after GRPO training.
 */
export async function loadTrajectory(industry: IndustryKey): Promise<Trajectory> {
  const key: IndustryKey = ["skincare", "food", "fitness"].includes(industry)
    ? industry
    : "skincare";
  try {
    const r = await fetch(`/trajectories/${key}.json`, { cache: "no-store" });
    if (r.ok) {
      const data = (await r.json()) as Trajectory;
      return data;
    }
  } catch {
    /* fall through to mock */
  }
  return TRAJECTORIES[key] || TRAJECTORIES.skincare;
}

export function pickTrajectory(industry: IndustryKey): Trajectory {
  return TRAJECTORIES[industry] || TRAJECTORIES.skincare;
}

export const TOOL_LABELS: Record<string, string> = {
  create_campaign: "Launch campaign",
  create_ad_set: "Create ad set",
  create_ad: "Create ad",
  get_metrics: "Pull metrics",
  update_budget: "Adjust budget",
  pause_ad: "Pause campaign",
  get_policy_updates: "Check policies",
  rewrite_creative: "Rewrite creative",
  noop: "Hold steady",
};
