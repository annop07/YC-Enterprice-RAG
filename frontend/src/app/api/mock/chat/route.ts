/**
 * Mock chat stream — stands in for `POST /chat` on the FastAPI service.
 *
 * It speaks the real SSE contract (session → sources → token* → done) at a
 * realistic pace, so the UI is exercising actual stream parsing, actual
 * out-of-order rendering and actual cancellation rather than a fake timer.
 * Swap it out by setting NEXT_PUBLIC_API_BASE.
 */
import { buildSources, pickAnswer } from "@/lib/mock-corpus";

const RETRIEVAL_MS = 420;
const FIRST_TOKEN_MS = 260;
const TOKEN_MS = 14;

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Split on whitespace but keep it, so re-joining reproduces the text exactly. */
function tokenize(text: string): string[] {
  return text.match(/\s+|[^\s]+/g) ?? [];
}

export async function POST(request: Request) {
  const body = (await request.json()) as {
    message?: string;
    session_id?: string;
  };
  const message = (body.message ?? "").trim();

  if (!message) {
    return Response.json({ detail: "message is required" }, { status: 422 });
  }

  const answer = pickAnswer(message);
  const sources = buildSources(answer);
  const started = Date.now();
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(frame(event, data)));
      };

      try {
        send("session", {
          session_id: body.session_id ?? `s_${Math.random().toString(36).slice(2, 10)}`,
          title: message.length > 56 ? `${message.slice(0, 56)}…` : message,
        });

        await sleep(RETRIEVAL_MS);
        if (request.signal.aborted) return;

        // Sources first: the citation cards should be on screen before the
        // answer starts arriving, not after it finishes.
        send("sources", {
          sources,
          candidates_considered: sources.length > 0 ? 20 : 0,
          retrieval_ms: RETRIEVAL_MS,
          // The mock speaks the same contract as the API, so the field is
          // present and null rather than absent — the demo has no embedding
          // model to be unable to read anything.
          notice: null,
        });

        await sleep(FIRST_TOKEN_MS);

        const tokens = tokenize(answer.answer);
        for (const token of tokens) {
          if (request.signal.aborted) return;
          send("token", { text: token });
          await sleep(TOKEN_MS);
        }

        const completion = Math.round(answer.answer.length / 3.6);
        send("done", {
          message_id: `m_${Math.random().toString(36).slice(2, 10)}`,
          latency_ms: Date.now() - started,
          usage: {
            prompt_tokens: 1180 + sources.length * 210,
            completion_tokens: completion,
            total_tokens: 1180 + sources.length * 210 + completion,
          },
          dropped_citations: 0,
          model: "qwen3.7-max",
        });
      } catch (e) {
        send("error", { detail: e instanceof Error ? e.message : "stream failed" });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Without this nginx buffers the whole body and the "stream" arrives
      // in one piece. Set here too so the mock behaves like the real thing.
      "X-Accel-Buffering": "no",
    },
  });
}
