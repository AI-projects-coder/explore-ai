// POST /api/ask -> forward the question to FastAPI as the logged-in user.
import { backendFetch, errorResponse } from "@/lib/server/api";
import { clearToken, getToken, isSameOrigin } from "@/lib/server/session";

export async function POST(request: Request) {
  if (!isSameOrigin(request)) return errorResponse(403, "Forbidden");

  const token = await getToken();
  if (!token) return errorResponse(401, "Your session has expired. Please log in again.");

  const body = await request.json().catch(() => null);
  const question = typeof body?.question === "string" ? body.question : "";

  let res: Response;
  try {
    // Only the question is forwarded: who the user is comes from the token, never from the browser.
    res = await backendFetch("/ask", { method: "POST", token, body: JSON.stringify({ question }) });
  } catch {
    return errorResponse(503, "The assistant is not reachable right now. Please try again.");
  }

  if (res.status === 401) {
    await clearToken();
    return errorResponse(401, "Your session has expired. Please log in again.");
  }
  if (res.status === 422) return errorResponse(422, "Please enter a question of up to 1,000 characters.");
  if (!res.ok) return errorResponse(503, "The assistant is temporarily unavailable. Please try again.");
  return Response.json(await res.json());
}
