import { HrAssistant } from "@/components/HrAssistant";
import { getSessionUser } from "@/lib/server/session";

// A Server Component: it reads the session cookie on the server, so the page renders already
// logged in (or not) without a flash. Only the display info (id, groups) reaches the browser.
export default async function Home() {
  const user = await getSessionUser();
  return <HrAssistant initialUser={user} />;
}
