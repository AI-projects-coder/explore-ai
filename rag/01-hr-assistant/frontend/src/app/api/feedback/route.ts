// POST /api/feedback -> forward a thumbs up/down for one answer to FastAPI.
import { backendFetch, errorResponse } from "@/lib/server/api";
import { getToken, isSameOrigin } from "@/lib/server/session";

export async function POST(request: Request) {
  if (!isSameOrigin(request)) return errorResponse(403, "Forbidden");

  const token = await getToken();
  if (!token) return errorResponse(401, "Your session has expired. Please log in again.");

  const body = await request.json().catch(() => null);
  const payload = {
    interaction_id: String(body?.interaction_id ?? ""),
    rating: body?.rating === "up" ? "up" : "down",
    comment: typeof body?.comment === "string" ? body.comment.slice(0, 1000) : null,
  };

  try {
    const res = await backendFetch("/feedback", { method: "POST", token, body: JSON.stringify(payload) });
    if (!res.ok) return errorResponse(res.status === 404 ? 404 : 502, "Feedback could not be saved.");
  } catch {
    return errorResponse(503, "Feedback could not be saved.");
  }
  return new Response(null, { status: 204 });
}
