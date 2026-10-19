// Shapes shared by the browser and the Next.js server. They mirror the FastAPI schemas.

export type SessionUser = {
  id: string;
  groups: string[];
  expiresAt: number; // unix seconds
};

export type Citation = {
  number: number;
  title: string;
  section: string;
  pages: string;
};

export type AskResponse = {
  answer: string;
  answerable: boolean;
  citations: Citation[];
  interaction_id?: string; // present once the backend logs interactions (needed for feedback)
};

export type FeedbackRating = "up" | "down";
