# Ingestion Pipeline

How text gets from a source into a searchable, citable chunk. Implemented in
`api/app/ingest/`.

## Connectors

A connector is the only component that knows where text comes from. Everything
downstream sees `RawDocument` and nothing else, so adding Confluence or Google
Drive is one new class rather than a second pipeline.

- `DirConnector` walks a directory for Markdown, skipping dot-directories and
  vendored trees such as `node_modules`. Without that skip list, pointing it at
  a project root indexes every dependency's README.
- `FileConnector` takes explicit paths, which is what the upload endpoint hands over.
- `PDFConnector` reads PDFs through PyMuPDF. A chunk never spans a page, so the
  page number on a citation is unambiguous. A file whose pages average fewer
  than forty characters is almost certainly a scan with no text layer, and the
  connector says so instead of indexing nothing.
- `GitHubConnector` takes a repository's whole layout from one recursive tree
  call — listing directory by directory burns a rate limit of sixty requests an
  hour on nothing — and builds permalinks from the commit SHA rather than the
  branch, so a link keeps pointing at the text that was indexed.

Each document carries a `source_id` that is stable within its source — a path
relative to the ingest root, not an absolute one. Ingesting the same corpus from
a different checkout updates the existing rows instead of duplicating them.

## Chunking

Two properties matter, and everything else follows from them.

**Chunk boundaries fall on line boundaries.** That is what makes `line_start`
and `line_end` exact, and an exact line range is what makes a citation
clickable rather than decorative. Blank lines at either edge of a chunk are
trimmed from the range as well as the text, because a range one line wider than
its content highlights the wrong first line.

**Token counts come from the embedding model's own tokenizer.** Counting
characters and dividing by four is off by roughly 30% on Thai, and the failure
mode is silent: the model truncates at its input ceiling and the tail of the
chunk is simply never embedded. The chunker takes the counter as a parameter,
so its tests run without loading a model.

The splitter is structure-aware. It tracks the Markdown heading stack to build
a breadcrumb for every line, ignores headings inside fenced code blocks, and
prefers to break before a heading once the current chunk is at least half full.
A single line too long to fit is split on token offsets; those pieces all report
the same line number, which is still true.

Chunks are 400 tokens with 80 tokens of overlap. Not the 500 to 1000 originally
planned: every multilingual model fastembed offers caps at 512 input tokens, and
the title and heading prefix has to fit inside the same budget. The pipeline
clamps `CHUNK_TOKENS` to what the model can actually encode and logs the clamp
rather than letting it be discovered later as missing text.

## Embedding

The text that gets embedded is not the text that gets displayed. Embedding input
is prefixed with the document title and heading breadcrumb — a cheap form of
contextual retrieval that costs no extra LLM call and stops short chunks, a bare
list or a code block, from losing every clue about what they belong to. Only the
chunk content appears in a citation.

Counting and embedding share one loaded model. fastembed leaves truncation
switched on, so counting through its tokenizer would report the ceiling for
anything longer and hide the exact overflow this is meant to detect; an
independent copy with truncation disabled counts honestly.

Models with asymmetric training, the e5 family, get `query:` and `passage:`
prefixes. bge and the others do not want them.

## Re-indexing

Every document carries a hash of its extracted text. Re-ingesting an unchanged
file is a no-op and costs no embedding calls. A changed file deletes all of its
chunks and re-inserts them inside one transaction, so a partially re-indexed
document is never visible to a search.

That hash covers the extracted text and nothing else. A file whose bytes are
identical but whose extracted *metadata* would now come out differently —
because the extraction code changed rather than the file — hashes the same and
is skipped with the status `unchanged`. That status is true of the bytes and
misleading about the outcome: the stored title stays whatever the previous
extractor produced. Improving title extraction and re-running an ordinary
ingest therefore changes nothing.

The consequence is not a stale label. The title is embedded rather than merely
displayed — `build_embed_input` prefixes it onto the chunk before the vector is
computed — so an out-of-date title sits inside the embeddings and skews what
those chunks match, while the citation card that shows only the chunk content
looks fine. Repairing the column by hand would fix what is displayed and leave
the vectors wrong. Re-index with `force` after changing anything in the
extraction path: `--force` on the CLI, `"force": true` on the ingest endpoints.
It re-derives and re-embeds a document whose hash still matches, which makes it
the only way to pick up an extractor change rather than merely a way to skip
the cache.

Changing the embedding model to one with a different width is a re-index, not a
config edit: the vector column width is fixed when the table is created, and the
embedder refuses to start if `EMBED_DIM` disagrees with what the model returns.
