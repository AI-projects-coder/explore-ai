import "server-only"; // build error if this file is ever imported into browser code

import { cookies } from "next/headers";

import type { SessionUser } from "@/lib/types";

// The JWT lives ONLY in this httpOnly cookie: browser JavaScript can never read it, so an XSS bug
// can't steal it. The Next.js server reads it and forwards it to FastAPI as a Bearer token.
const COOKIE = "hr_session";

export async function getToken(): Promise<string | undefined> {
  return (await cookies()).get(COOKIE)?.value;
}

export async function setToken(token: string, maxAgeSeconds: number): Promise<void> {
  (await cookies()).set(COOKIE, token, {
    httpOnly: true, // not readable from JavaScript
    secure: process.env.NODE_ENV === "production", // HTTPS-only in production
    sameSite: "strict", // never sent on requests started by other sites (CSRF protection)
    path: "/",
    maxAge: maxAgeSeconds, // the cookie disappears when the token expires
  });
}

export async function clearToken(): Promise<void> {
  (await cookies()).delete(COOKIE);
}

/**
 * Read who is logged in, FOR DISPLAY ONLY. We don't verify the signature here: the FastAPI backend
 * verifies it on every request, and that check is the one that protects data.
 */
export async function getSessionUser(): Promise<SessionUser | null> {
  const token = await getToken();
  return token ? userFromToken(token) : null;
}

export function userFromToken(token: string): SessionUser | null {
  try {
    const payload = JSON.parse(Buffer.from(token.split(".")[1], "base64url").toString("utf8"));
    if (typeof payload.exp !== "number" || payload.exp * 1000 < Date.now()) return null;
    return { id: String(payload.sub), groups: payload.groups ?? [], expiresAt: payload.exp };
  } catch {
    return null;
  }
}

/** Second CSRF layer (after sameSite): state-changing requests must come from our own pages. */
export function isSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  return origin !== null && origin === new URL(request.url).origin;
}
