"use client";

import { useState } from "react";

import type { SessionUser } from "@/lib/types";

// Local development only: these mirror DEV_USERS in the backend (core/auth.py).
// In production this screen is replaced by the company's real sign-in (e.g. Google).
const DEV_USERS = [
  { id: "intern_ravi", name: "Ravi", role: "Intern", groups: ["employee"] },
  { id: "manager_priya", name: "Priya", role: "Engineering manager", groups: ["employee", "manager"] },
  { id: "hr_anita", name: "Anita", role: "HR partner", groups: ["employee", "hr"] },
  { id: "vendor_sam", name: "Sam", role: "Contractor", groups: ["contractor"] },
];

export function LoginScreen({ notice, onLogin }: { notice: string | null; onLogin: (u: SessionUser) => void }) {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function login(userId: string) {
    setPending(userId);
    setError(null);
    try {
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Login failed");
      onLogin(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setPending(null);
    }
  }

  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <h1 className="text-2xl font-semibold tracking-tight">HR Policy Assistant</h1>
        <p className="mt-2 text-sm text-muted">
          Ask about leave, working from home, conduct and other Kalpa Systems policies. Every answer shows its
          sources.
        </p>

        {notice && (
          <p className="mt-6 rounded-lg bg-notice-soft px-4 py-3 text-sm text-notice" role="status">
            {notice}
          </p>
        )}

        <div className="mt-8 rounded-xl border border-border bg-surface p-2">
          <p className="px-3 pt-2 pb-3 text-xs font-medium uppercase tracking-wide text-muted">
            Sign in as a test user (local only)
          </p>
          <ul className="flex flex-col gap-1">
            {DEV_USERS.map((u) => (
              <li key={u.id}>
                <button
                  type="button"
                  onClick={() => login(u.id)}
                  disabled={pending !== null}
                  className="flex w-full items-center justify-between gap-4 rounded-lg px-3 py-3 text-left transition-colors hover:bg-surface-muted disabled:opacity-60"
                >
                  <span>
                    <span className="block font-medium">
                      {u.name} <span className="font-normal text-muted">· {u.role}</span>
                    </span>
                    <span className="mt-1 flex flex-wrap gap-1">
                      {u.groups.map((g) => (
                        <span key={g} className="rounded bg-accent-soft px-1.5 py-0.5 font-mono text-xs text-accent">
                          {g}
                        </span>
                      ))}
                    </span>
                  </span>
                  <span className="text-sm text-muted">{pending === u.id ? "Signing in…" : "→"}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {error && (
          <p className="mt-4 rounded-lg bg-danger-soft px-4 py-3 text-sm text-danger" role="alert">
            {error}
          </p>
        )}
      </div>
    </main>
  );
}
