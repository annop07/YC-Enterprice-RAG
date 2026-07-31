import { documentSummaries } from "@/lib/mock-corpus";

export async function GET() {
  return Response.json(documentSummaries());
}
