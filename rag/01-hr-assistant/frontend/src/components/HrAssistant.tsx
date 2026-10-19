"use client";

import { useState } from "react";

import { Chat } from "@/components/Chat";
import { LoginScreen } from "@/components/LoginScreen";
import type { SessionUser } from "@/lib/types";

export function HrAssistant({ initialUser }: { initialUser: SessionUser | null }) {
  const [user, setUser] = useState<SessionUser | null>(initialUser);
  const [notice, setNotice] = useState<string | null>(null);

  if (!user) {
    return (
      <LoginScreen
        notice={notice}
        onLogin={(u) => {
          setNotice(null);
          setUser(u);
        }}
      />
    );
  }

  return (
    <Chat
      key={user.id} // a different user gets a fresh, empty conversation
      user={user}
      onLogout={async () => {
        await fetch("/api/session", { method: "DELETE" });
        setUser(null);
      }}
      onSessionExpired={() => {
        setNotice("Your session expired. Please log in again.");
        setUser(null);
      }}
    />
  );
}
