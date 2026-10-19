"use client";

import { useState } from "react";

import type { AskResponse, FeedbackRating } from "@/lib/types";

/** Turn "up to 4 days [1][3]." into text with clickable citation markers. */
function AnswerText({ text, onCite, active }: { text: string; onCite: (n: number) => void; active: number | null }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return <span key={i}>{part}</span>;
        const n = Number(match[1]);
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCite(n)}
            aria-label={`Show source ${n}`}
            className={`mx-0.5 inline-flex h-5 min-w-5 -translate-y-0.5 items-center justify-center rounded px-1 align-middle font-mono text-[11px] transition-colors ${
              active === n ? "bg-accent text-accent-foreground" : "bg-accent-soft text-accent hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {n}
          </button>
        );
      })}
    </p>
  );
}

function Feedback({ interactionId }: { interactionId: string }) {
  const [state, setState] = useState<"idle" | "comment" | "sending" | "sent" | "error">("idle");
  const [comment, setComment] = useState("");

  async function send(rating: FeedbackRating, text?: string) {
    setState("sending");
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ interaction_id: interactionId, rating, comment: text ?? null }),
    }).catch(() => null);
    setState(res?.ok ? "sent" : "error");
  }

  if (state === "sent") return <p className="text-xs text-muted">Thanks for the feedback.</p>;
  if (state === "error") return <p className="text-xs text-danger">Feedback could not be saved.</p>;
  if (state === "comment") {
    return (
      <form
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          send("down", comment);
        }}
      >
        <input
          autoFocus
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          maxLength={1000}
          placeholder="What was wrong? (optional)"
          className="flex-1 rounded-md border border-border bg-surface px-3 py-1.5 text-sm outline-none focus:border-accent"
        />
        <button type="submit" className="rounded-md bg-accent px-3 py-1.5 text-sm text-accent-foreground">
          Send
        </button>
      </form>
    );
  }
  return (
    <div className="flex items-center gap-1 text-xs text-muted">
      <span className="mr-1">Was this helpful?</span>
      <button
        type="button"
        disabled={state === "sending"}
        onClick={() => send("up")}
        className="rounded px-2 py-1 hover:bg-surface-muted"
      >
        Yes
      </button>
      <button
        type="button"
        disabled={state === "sending"}
        onClick={() => setState("comment")}
        className="rounded px-2 py-1 hover:bg-surface-muted"
      >
        No
      </button>
    </div>
  );
}

export function AnswerCard({ answer }: { answer: AskResponse }) {
  const [active, setActive] = useState<number | null>(null);

  if (!answer.answerable) {
    // Not found / off topic: shown calmly and clearly different from a real answer.
    return (
      <div className="rounded-xl border border-border bg-surface-muted px-4 py-3 text-sm text-muted">
        <p className="leading-relaxed">{answer.answer}</p>
        {answer.interaction_id && (
          <div className="mt-3">
            <Feedback interactionId={answer.interaction_id} />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3">
      <AnswerText text={answer.answer} onCite={(n) => setActive(active === n ? null : n)} active={active} />

      {answer.citations.length > 0 && (
        <ol className="mt-4 flex flex-col gap-1.5 border-t border-border pt-3" aria-label="Sources">
          {answer.citations.map((c) => (
            <li key={c.number}>
              <button
                type="button"
                onClick={() => setActive(active === c.number ? null : c.number)}
                className={`flex w-full items-start gap-2.5 rounded-lg px-2 py-1.5 text-left text-sm transition-colors ${
                  active === c.number ? "bg-accent-soft" : "hover:bg-surface-muted"
                }`}
              >
                <span
                  className={`mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 font-mono text-[11px] ${
                    active === c.number ? "bg-accent text-accent-foreground" : "bg-accent-soft text-accent"
                  }`}
                >
                  {c.number}
                </span>
                <span className="min-w-0">
                  <span className="font-medium">{c.title}</span>
                  <span className="text-muted">
                    {" "}
                    · {c.section}
                    {c.pages && ` · ${c.pages}`}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}

      {answer.interaction_id && (
        <div className="mt-3 border-t border-border pt-2">
          <Feedback interactionId={answer.interaction_id} />
        </div>
      )}
    </div>
  );
}
