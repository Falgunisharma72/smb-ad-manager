"use client";

import { animate, useMotionValue, useTransform, motion } from "framer-motion";
import { useEffect } from "react";

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  formatter?: (n: number) => string;
  className?: string;
}

/** Smoothly counts up from 0 to `value` on mount. */
export function AnimatedCounter({
  value,
  duration = 1.2,
  formatter = (n) => Math.round(n).toLocaleString("en-IN"),
  className,
}: AnimatedCounterProps) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => formatter(v));

  useEffect(() => {
    const controls = animate(count, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
    });
    return () => controls.stop();
  }, [value, duration, count]);

  return <motion.span className={className}>{rounded}</motion.span>;
}
