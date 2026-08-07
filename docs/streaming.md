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
