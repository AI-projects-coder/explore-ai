// POST /api/ask -> forward the question to FastAPI as the logged-in user.
import { backendFetch, errorResponse, newRequestId } from "@/lib/server/api";
import { clearToken, getToken, isSameOrigin } from "@/lib/server/session";

export async function POST(request: Request) {
  if (!isSameOrigin(request)) return errorResponse(403, "Forbidden");

  const token = await getToken();
  if (!token) return errorResponse(401, "Your session has expired. Please log in again.");

  const body = await request.json().catch(() => null);
  const question = typeof body?.question === "string" ? body.question : "";

  const requestId = newRequestId();
  let res: Response;
  try {
    // Only the question is forwarded: who the user is comes from the token, never from the browser.
    res = await backendFetch("/ask", {
      method: "POST",
      token,
      body: JSON.stringify({ question }),
      headers: { "x-request-id": requestId },
    });
  } catch {
    return errorResponse(503, "The assistant is not reachable right now. Please try again.", requestId);
  }

  if (res.status === 401) {
    await clearToken();
    return errorResponse(401, "Your session has expired. Please log in again.");
  }
  if (res.status === 422) return errorResponse(422, "Please enter a question of up to 1,000 characters.");
  if (res.status === 429) {
    const seconds = Number(res.headers.get("retry-after") ?? "60");
    const minutes = Math.ceil(seconds / 60);
    const hours = Math.ceil(seconds / 3600);
    const wait =
      seconds <= 90 ? "a minute" // the per-minute limit
      : minutes < 60 ? `${minutes} minutes` // the daily limit, shortly before midnight
      : hours <= 3 ? `${hours} hour${hours > 1 ? "s" : ""}`
      : "tomorrow"; // the daily limit: resets at midnight (India time)
    return errorResponse(429, `You've asked a lot of questions. Please try again in ${wait}.`);
  }
  if (!res.ok) return errorResponse(503, "The assistant is temporarily unavailable. Please try again.", requestId);
  return Response.json(await res.json());
}
