"use client";

import { useEffect } from "react";

/**
 * Global haptic provider.
 * Fires a short vibration on any click that targets a button, link, or
 * known interactive surface (.card-surface / .stat-card). Silently no-ops
 * on devices without the Vibration API (i.e., desktop, iOS Safari).
 */
export function HapticProvider() {
  useEffect(() => {
    if (typeof navigator === "undefined" || !("vibrate" in navigator)) return;

    const SELECTOR =
      "button, a, .card-surface, .stat-card, [data-haptic]";

    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const interactive = target.closest(SELECTOR);
      if (!interactive) return;

      // Stronger pulse for primary buttons, lighter for everything else
      const isPrimary =
        interactive.classList.contains("btn-primary") ||
        interactive.tagName === "BUTTON" &&
          interactive.querySelector(".btn-primary");
      try {
        navigator.vibrate(isPrimary ? [12] : [8]);
      } catch {
        /* unsupported - ignore */
      }
    };

    document.addEventListener("click", handler, { passive: true });
    return () => document.removeEventListener("click", handler);
  }, []);

  return null;
}
