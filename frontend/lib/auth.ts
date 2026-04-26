/**
 * Simple cookie-based auth for hackathon demo.
 * Not meant to be secure - just a gate so only judges with credentials can see the demo.
 *
 * Username / password live in environment variables:
 *   AUTH_USERNAME    (default: admin)
 *   AUTH_PASSWORD    (default: hackathon2026)
 */
import { cookies } from "next/headers";

export const AUTH_COOKIE = "smb_auth";
const EXPECTED_TOKEN = "ok-v1";

export function getExpectedCredentials() {
  return {
    username: process.env.AUTH_USERNAME || "admin",
    password: process.env.AUTH_PASSWORD || "hackathon2026",
  };
}

export async function isAuthenticated(): Promise<boolean> {
  const store = await cookies();
  return store.get(AUTH_COOKIE)?.value === EXPECTED_TOKEN;
}

export const AUTH_TOKEN = EXPECTED_TOKEN;
