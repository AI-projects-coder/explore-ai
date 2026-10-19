import "server-only";

const API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";

/** Call the FastAPI backend from the Next.js server, optionally as the logged-in user. */
export async function backendFetch(path: string, init: RequestInit & { token?: string } = {}): Promise<Response> {
  const { token, headers, ...rest } = init;
  return fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    cache: "no-store", // answers are per user: never cache them
    signal: AbortSignal.timeout(60_000), // a hung backend must not hang the page forever
  });
}

/** A JSON error the browser can show, without leaking backend internals. */
export function errorResponse(status: number, message: string): Response {
  return Response.json({ error: message }, { status });
}
