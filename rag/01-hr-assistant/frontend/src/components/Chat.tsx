"use client";

import { useEffect, useRef, useState } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import type { AskResponse, SessionUser } from "@/lib/types";

const MAX_CHARS = 1000; // same limit as the API (api/schemas.py)

// Example questions per test user, chosen to show both answers AND access control in action.
const SUGGESTIONS: Record<string, string[]> = {
  intern_ravi: [
    "How many casual leaves can I carry forward?",
    "Can I work from Dubai for a month?",
    "What is the maximum salary for an L4 lead engineer?",
  ],
  manager_priya: [
    "What is the maximum salary for an L4 lead engineer?",
    "How should I calibrate ratings with my team?",
    "My report got a rating of 1. What happens next?",
  ],
  hr_anita: [
    "What is the minimum salary for a Director?",
    "What is the rating distribution guideline?",
    "How many sick leaves do employees get?",
  ],
  vendor_sam: ["How many casual leaves can I carry forward?", "What are the core hours?"],
};

type NewTurn =
  | { kind: "question"; text: string }
  | { kind: "answer"; answer: AskResponse }
  | { kind: "error"; text: string; question: string; reference?: string };
type Turn = NewTurn & { id: number };

export function Chat({
  user,
  onLogout,
  onSessionExpired,
}: {
  user: SessionUser;
  onLogout: () => void;
  onSessionExpired: () => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const nextId = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, loading]);

  function add(turn: NewTurn) {
    const id = nextId.current++;
    setTurns((prev) => [...prev, { ...turn, id }]);
  }

  async function ask(question: string) {
    const q = question.trim();
    if (!q || loading) return;
    setInput("");
    add({ kind: "question", text: q });
    setLoading(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.status === 401) return onSessionExpired();
      if (!res.ok) {
        add({ kind: "error", text: data.error ?? "Something went wrong. Please try again.", question: q, reference: data.reference });
        return;
      }
      add({ kind: "answer", answer: data as AskResponse });
    } catch {
      add({ kind: "error", text: "Something went wrong. Please check your connection and try again.", question: q });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <h1 className="font-semibold tracking-tight">HR Policy Assistant</h1>
          <div className="flex items-center gap-3 text-sm">
            <span className="hidden text-muted sm:inline">
              {user.id}{" "}
              {user.groups.map((g) => (
                <span key={g} className="ml-1 rounded bg-accent-soft px-1.5 py-0.5 font-mono text-xs text-accent">
                  {g}
                </span>
              ))}
            </span>
            <button type="button" onClick={onLogout} className="rounded-md px-2 py-1 text-muted hover:bg-surface-muted">
              Log out
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-6" aria-live="polite">
          {turns.length === 0 && (
            <div className="py-10">
              <p className="text-lg font-medium">Ask a question about HR policies</p>
              <p className="mt-1 text-sm text-muted">
                You see answers only from policies your groups can access. Try one of these:
              </p>
              <div className="mt-5 flex flex-col gap-2">
                {(SUGGESTIONS[user.id] ?? []).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => ask(s)}
                    className="rounded-lg border border-border bg-surface px-4 py-2.5 text-left text-sm transition-colors hover:border-accent"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((t) =>
            t.kind === "question" ? (
              <div key={t.id} className="flex justify-end">
                <p className="max-w-[85%] rounded-xl bg-accent px-4 py-2.5 text-accent-foreground whitespace-pre-wrap">
                  {t.text}
                </p>
              </div>
            ) : t.kind === "answer" ? (
              <div key={t.id} className="max-w-[92%]">
                <AnswerCard answer={t.answer} />
              </div>
            ) : (
              <div key={t.id} className="max-w-[92%] rounded-xl bg-danger-soft px-4 py-3 text-sm text-danger" role="alert">
                <p>{t.text}</p>
                {t.reference && <p className="mt-1 font-mono text-xs opacity-80">Reference: {t.reference}</p>}
                <button type="button" onClick={() => ask(t.question)} className="mt-2 font-medium underline">
                  Try again
                </button>
              </div>
            ),
          )}

          {loading && (
            <p className="text-sm text-muted" role="status">
              Searching the policies…
            </p>
          )}
          <div ref={bottom} />
        </div>
      </main>

      <footer className="border-t border-border bg-surface">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2 px-4 py-3"
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Your question
          </label>
          <textarea
            id="question"
            rows={1}
            value={input}
            maxLength={MAX_CHARS}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(input);
              }
            }}
            placeholder="e.g. How many sick leaves do I get?"
            className="max-h-40 min-h-11 flex-1 resize-none rounded-lg border border-border bg-background px-3 py-2.5 outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="h-11 rounded-lg bg-accent px-4 font-medium text-accent-foreground disabled:opacity-50"
          >
            Ask
          </button>
        </form>
        {input.length > MAX_CHARS * 0.8 && (
          <p className="mx-auto max-w-3xl px-4 pb-2 text-right text-xs text-muted">
            {input.length}/{MAX_CHARS}
          </p>
        )}
      </footer>
    </div>
  );
}
