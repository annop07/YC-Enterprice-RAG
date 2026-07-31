import { CORPUS_STATS } from "@/lib/mock-corpus";

export async function GET() {
  return Response.json(CORPUS_STATS);
}
