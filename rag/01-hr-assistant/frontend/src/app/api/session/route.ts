// GET    /api/session  -> who is logged in (or 401)
// POST   /api/session  -> log in as a dev user: FastAPI issues the JWT, we store it in the httpOnly cookie
// DELETE /api/session  -> log out
import { backendFetch, errorResponse } from "@/lib/server/api";
import { clearToken, getSessionUser, isSameOrigin, setToken, userFromToken } from "@/lib/server/session";

export async function GET() {
  const user = await getSessionUser();
  return user ? Response.json(user) : errorResponse(401, "Not logged in");
}

export async function POST(request: Request) {
  if (!isSameOrigin(request)) return errorResponse(403, "Forbidden");

  const body = await request.json().catch(() => null);
  const userId = typeof body?.user_id === "string" ? body.user_id : "";

  let res: Response;
  try {
    res = await backendFetch("/auth/dev-token", { method: "POST", body: JSON.stringify({ user_id: userId }) });
  } catch {
    return errorResponse(503, "The backend is not reachable. Is the API running?");
  }
  if (res.status === 404) return errorResponse(404, "Dev login is disabled in this environment");
  if (!res.ok) return errorResponse(400, "Unknown user");

  const { access_token, expires_in } = await res.json();
  await setToken(access_token, expires_in);
  return Response.json(userFromToken(access_token));
}

export async function DELETE(request: Request) {
  if (!isSameOrigin(request)) return errorResponse(403, "Forbidden");
  await clearToken();
  return new Response(null, { status: 204 });
}
