import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SMB Ad Manager — RL Training Environment",
  description:
    "Reward-hardened RL environment for LLM agents learning to manage Meta Ads for small businesses.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
