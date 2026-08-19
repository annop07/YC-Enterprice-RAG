import { deleteSession, getSession } from "@/lib/mock-sessions";

export const dynamic = "force-dynamic";

// Next 16: `params` is a Promise — synchronous access was removed in this major.
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const session = getSession(id);
  if (!session) return Response.json({ detail: "not found" }, { status: 404 });
  return Response.json(session);
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  // 204 and a bare 404, the same as `DELETE /sessions/{id}` on the API.
  if (!deleteSession(id)) return Response.json({ detail: "not found" }, { status: 404 });
  return new Response(null, { status: 204 });
}
