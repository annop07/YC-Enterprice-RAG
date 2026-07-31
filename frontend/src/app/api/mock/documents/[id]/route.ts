import { documentText } from "@/lib/mock-corpus";

// Next 16: `params` is a Promise — synchronous access was removed in this major.
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const doc = documentText(id);
  if (!doc) return Response.json({ detail: "not found" }, { status: 404 });
  return Response.json(doc);
}
