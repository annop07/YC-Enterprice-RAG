import { listSessions } from "@/lib/mock-sessions";

// The store lives in process memory and changes with every turn, so this
// handler must run per request. Without it the build prerenders the list once
// — empty, forever — and `npm run build && npm start` ships a sidebar that
// never fills up.
export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json(listSessions());
}
