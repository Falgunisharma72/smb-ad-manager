import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { AUTH_COOKIE, AUTH_TOKEN, getExpectedCredentials } from "@/lib/auth";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { username, password } = body as { username?: string; password?: string };
    const expected = getExpectedCredentials();

    if (username !== expected.username || password !== expected.password) {
      return NextResponse.json(
        { ok: false, message: "Invalid credentials" },
        { status: 401 }
      );
    }

    const store = await cookies();
    store.set(AUTH_COOKIE, AUTH_TOKEN, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      maxAge: 60 * 60 * 12,
      path: "/",
    });

    return NextResponse.json({ ok: true });
  } catch {
    return NextResponse.json({ ok: false, message: "Bad request" }, { status: 400 });
  }
}
