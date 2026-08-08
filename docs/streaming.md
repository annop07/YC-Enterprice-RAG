# Streaming and the SSE contract

## Event order

```
event: session   {"session_id": "...", "title": "..."}
event: sources   {"sources": [...], "candidates_considered": 20, "retrieval_ms": 420}
event: token     {"text": "Retrieval "}
event: done      {"latency_ms": 6210, "usage": {...}, "dropped_citations": 0, "model": "..."}
```

The order is the point. Sources are flushed before the first token so the
citation cards are on screen and readable while the answer is still being
written, instead of appearing after it.

`EventSource` is not used on the client. It is GET-only and cannot carry a
request body, so the stream is read off a `fetch` POST and the SSE framing is
parsed by hand. The response carries `X-Accel-Buffering: no`; without it a proxy
buffers the whole body and the answer lands in one lump. The stream still works,
it just stops looking like one.

## What `dropped_citations` counts

The line under a finished answer reading **"0 citations dropped"** is the
`dropped_citations` field of the `done` event. It is the number of citation
markers the model made up and the guard removed.

The prompt gives the model a numbered list of retrieved context blocks — five of
them at the default `top_k`, numbered 1 to 5 — and asks it to mark every factual
claim with the number of the block it came from. Nothing stops it citing a
seventh block that was never supplied. `strip_unsupported_citations` in
`api/app/chat/prompt.py` scans the finished answer for `[n]` markers and deletes
each one whose number is not among the sources actually retrieved for that turn,
counting as it goes. That is the whole condition: the marker is a bracketed
integer, and that integer is not one of the blocks the model was shown. A marker
pointing at a block that does exist is never removed, however badly the sentence
attached to it misreads the source.

The test is textual and knows nothing about context. A bracketed number the
model is merely quoting — an example of a bad citation, a version, an array
index — is stripped exactly like a fabricated one, and counted. Documents in
this corpus avoid writing bracketed integers in prose for that reason: the
prompt tells the model to prefer the documents' own wording, so anything written
that way here comes back in an answer and is then deleted from it.

Zero means the model invented no citation numbers. It does not mean every claim
in the answer is supported. The system prompt says a claim with no number is
treated as unsupported, but that is an instruction to the model rather than a
check on its output — an answer carrying no citations at all reports zero
dropped citations, because there was nothing bracketed to remove. Read the
number as a fabrication count, not as a coverage score.

A non-zero count means that many invented markers were stripped, and the answer
kept in history is the stripped version, with the leftover space before any
trailing punctuation tidied up. The tokens had already gone to the browser by
the time the whole answer existed to be checked, so the live stream shows the
bad marker and the stored answer does not. The count is the part that tells you
it happened.

## Choosing a model to stream from

Measured against the KKU IntelSphere proxy. The number that matters for a chat
UI is time to first *visible* token, not time to first chunk.

| model | first visible token | total | completion tokens |
| --- | --- | --- | --- |
| qwen3-next-80b-a3b-instruct | 0.9s | 1.1s | 72 |
| claude-haiku-4.5 | 1.7s | 3.1s | 118 |
| gemini-3.6-flash | 2.4s | 2.4s | 196 |
| qwen3.7-max | 18.2s | 20.7s | 940 |

`qwen3.7-max` is a reasoning model. It streams, but it spends about 3,500
characters of reasoning content before emitting a single visible token, and a
short answer costs an order of magnitude more completion tokens. It is a fine
choice for batch structured extraction and a bad one here.

`gemini-3.6-flash` returns its whole answer in a single content delta. It
streams on paper and looks buffered on screen.

The default is `qwen3-next-80b-a3b-instruct`. `claude-haiku-4.5` is the
alternative when answer quality matters more than the first second.

Both `stream=True` and `stream_options={"include_usage": True}` work through the
proxy, so the `done` event carries real token counts rather than an estimate.
