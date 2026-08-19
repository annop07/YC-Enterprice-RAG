"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  IconAlertTriangle,
  IconBrandGithub,
  IconFileTypePdf,
  IconLoader2,
  IconMarkdown,
  IconPaperclip,
  IconTrash,
  IconUpload,
  IconX,
} from "@tabler/icons-react";
import {
  IS_DEMO,
  deleteDocument,
  getDocuments,
  getJob,
  ingestFiles,
  ingestGitHub,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type {
  DocumentSummary,
  IngestJob,
  IngestResponse,
  IngestResult,
  SourceType,
} from "@/lib/types";

const ICONS: Record<SourceType, typeof IconMarkdown> = {
  markdown: IconMarkdown,
  pdf: IconFileTypePdf,
  github: IconBrandGithub,
};

const ACCEPT = ".md,.mdx,.markdown,.pdf";

//: Often enough to look live, rarely enough that a long ingest is not a
//: request per file. The job row is cheap to read; the work is not.
const POLL_MS = 600;

/**
 * What is in the index, and how to put more in it.
 *
 * The API has taken uploads and repositories since the ingestion layer landed;
 * until now the only way to reach it was curl, which makes "ingest your
 * documents" a claim about the backend rather than something a user can do.
 */
export function CorpusPanel({
  open,
  onClose,
  onIngested,
}: {
  open: boolean;
  onClose: () => void;
  onIngested: () => void;
}) {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null);
  const [repo, setRepo] = useState("");
  const [prefix, setPrefix] = useState("");
  const [busy, setBusy] = useState<"files" | "github" | null>(null);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    getDocuments()
      .then(setDocuments)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "could not list documents"),
      );
  }, []);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  /**
   * Start a job and follow it to the end.
   *
   * The endpoint answers before the work is done, so the panel polls rather
   * than waits. That is the whole reason indexing became a job: a repository
   * takes longer to read than a browser will hold a request open, and one
   * spinner for the entire batch could not say which file it was on.
   */
  async function run(kind: "files" | "github", start: () => Promise<IngestJob>) {
    setBusy(kind);
    setError(null);
    setJob(null);
    try {
      let current = await start();
      setJob(current);

      while (current.status === "running") {
        await new Promise((r) => setTimeout(r, POLL_MS));
        current = await getJob(current.id);
        setJob(current);
        // The corpus counts move while it runs, not only at the end.
        onIngested();
      }

      if (current.status === "failed") {
        setError(current.error ?? "the job failed");
      }
      refresh();
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : "ingest failed");
    } finally {
      setBusy(null);
    }
  }

  function upload(files: FileList | File[] | null) {
    const list = Array.from(files ?? []);
    if (list.length === 0) return;
    void run("files", () => ingestFiles(list));
  }

  if (!open) return null;

  return (
    <>
      <button
        aria-label="Close corpus"
        className="fixed inset-0 z-40 bg-foreground/10 backdrop-blur-[1px]"
        onClick={onClose}
        tabIndex={-1}
        type="button"
      />

      <aside
        aria-label="Corpus"
        className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-border border-l bg-background shadow-xl"
      >
        <header className="flex items-start gap-3 border-border border-b px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 className="font-heading font-semibold text-foreground text-sm tracking-tight">
              Corpus
            </h2>
            <p className="mt-1 text-muted-foreground text-xs">
              {documents
                ? `${documents.length} documents · ${documents.reduce((n, d) => n + d.chunk_count, 0)} chunks`
                : "Loading…"}
            </p>
          </div>
          <Button
            aria-label="Close"
            className="rounded-md"
            onClick={onClose}
            size="icon-sm"
            variant="ghost"
          >
            <IconX size={16} />
          </Button>
        </header>

        <div className="scroll-thin flex-1 space-y-6 overflow-y-auto px-5 py-5">
          {IS_DEMO && (
            <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/50 px-3 py-2 text-muted-foreground text-xs">
              <IconAlertTriangle className="mt-0.5 shrink-0" size={14} />
              {/* The text is its own block: inline <code> inside the flex row
                  becomes a flex item and breaks the sentence into columns. */}
              <p className="leading-5">
                This is the built-in demo corpus. Point{" "}
                <code className="rounded-[4px] bg-muted px-1 font-mono">
                  NEXT_PUBLIC_API_BASE
                </code>{" "}
                at the API to index documents of your own.
              </p>
            </div>
          )}

          <Uploader
            busy={busy === "files"}
            disabled={IS_DEMO}
            dragging={dragging}
            fileInput={fileInput}
            onDrop={upload}
            onDraggingChange={setDragging}
          />

          <GitHubForm
            busy={busy === "github"}
            disabled={IS_DEMO}
            onSubmit={() =>
              run("github", () =>
                ingestGitHub({
                  repo: repo.trim(),
                  path_prefix: prefix.trim() || undefined,
                }),
              )
            }
            prefix={prefix}
            repo={repo}
            setPrefix={setPrefix}
            setRepo={setRepo}
          />

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-destructive text-xs">
              <IconAlertTriangle className="mt-0.5 shrink-0" size={14} />
              {error}
            </p>
          )}

          {job && job.status === "running" && <Progress job={job} />}
          {job?.report && <Report report={job.report} />}

          <section className="space-y-2">
            <h3 className="font-medium text-foreground text-xs">Indexed</h3>
            {documents === null ? (
              <p className="text-muted-foreground text-xs">Loading…</p>
            ) : documents.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                Nothing indexed yet. Upload a file or point it at a repository.
              </p>
            ) : (
              <ul className="divide-y divide-border overflow-hidden rounded-xl ring-1 ring-foreground/10">
                {documents.map((doc) => (
                  <DocumentRow
                    doc={doc}
                    key={doc.id}
                    onDelete={
                      IS_DEMO
                        ? undefined
                        : () =>
                            void deleteDocument(doc.id)
                              .then(() => {
                                refresh();
                                onIngested();
                              })
                              .catch((e: unknown) =>
                                setError(
                                  e instanceof Error ? e.message : "delete failed",
                                ),
                              )
                    }
                  />
                ))}
              </ul>
            )}
          </section>
        </div>
      </aside>
    </>
  );
}

function Uploader({
  disabled,
  busy,
  dragging,
  onDraggingChange,
  onDrop,
  fileInput,
}: {
  disabled: boolean;
  busy: boolean;
  dragging: boolean;
  onDraggingChange: (value: boolean) => void;
  onDrop: (files: FileList | File[] | null) => void;
  fileInput: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <section className="space-y-2">
      <h3 className="font-medium text-foreground text-xs">Upload</h3>
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-4 py-8 text-center transition-colors",
          dragging && !disabled ? "border-ring bg-accent/60" : "border-border",
          disabled && "opacity-50",
        )}
        onDragLeave={(e) => {
          e.preventDefault();
          onDraggingChange(false);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) onDraggingChange(true);
        }}
        onDrop={(e) => {
          e.preventDefault();
          onDraggingChange(false);
          if (!disabled && !busy) onDrop(e.dataTransfer.files);
        }}
      >
        <input
          accept={ACCEPT}
          className="sr-only"
          disabled={disabled || busy}
          multiple
          onChange={(e) => {
            onDrop(e.target.files);
            e.target.value = ""; // so the same file can be re-uploaded
          }}
          ref={fileInput}
          type="file"
        />
        {busy ? (
          <span className="flex items-center gap-2 text-muted-foreground text-sm">
            <IconLoader2 className="animate-spin motion-reduce:animate-none" size={14} />
            Chunking and embedding…
          </span>
        ) : (
          <>
            <IconUpload aria-hidden className="text-muted-foreground" size={20} stroke={1.5} />
            <p className="text-muted-foreground text-xs">
              Drop Markdown or PDF here
            </p>
            <Button
              className="rounded-full"
              disabled={disabled}
              onClick={() => fileInput.current?.click()}
              size="sm"
              type="button"
              variant="outline"
            >
              <IconPaperclip size={14} />
              Choose files
            </Button>
          </>
        )}
      </div>
    </section>
  );
}

function GitHubForm({
  repo,
  setRepo,
  prefix,
  setPrefix,
  onSubmit,
  disabled,
  busy,
}: {
  repo: string;
  setRepo: (v: string) => void;
  prefix: string;
  setPrefix: (v: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  busy: boolean;
}) {
  // "owner/name" — the same shape the API validates, checked here so an obvious
  // typo does not need a round trip to be told about.
  const valid = /^[^/\s]+\/[^/\s]+$/.test(repo.trim());

  return (
    <section className="space-y-2">
      <h3 className="font-medium text-foreground text-xs">From GitHub</h3>
      <form
        className="space-y-2 rounded-xl p-3 ring-1 ring-foreground/10"
        onSubmit={(e) => {
          e.preventDefault();
          if (valid && !disabled && !busy) onSubmit();
        }}
      >
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            aria-label="Repository"
            className="font-mono text-xs"
            disabled={disabled || busy}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="owner/name"
            value={repo}
          />
          <Input
            aria-label="Path prefix"
            className="font-mono text-xs sm:max-w-40"
            disabled={disabled || busy}
            onChange={(e) => setPrefix(e.target.value)}
            placeholder="docs (optional)"
            value={prefix}
          />
        </div>
        <div className="flex items-center gap-2">
          <p className="text-muted-foreground text-[11px]">
            Markdown only, pinned to the current commit.
          </p>
          <Button
            className="ml-auto rounded-md"
            disabled={disabled || busy || !valid}
            size="sm"
            type="submit"
          >
            {busy ? (
              <IconLoader2 className="animate-spin motion-reduce:animate-none" size={14} />
            ) : (
              <IconBrandGithub size={14} />
            )}
            Index
          </Button>
        </div>
      </form>
    </section>
  );
}

//: What each stage is called on screen. A job spends most of its life in one
//: of three, and "working" for all of them is what this replaced.
const PHASES: Record<string, string> = {
  reading: "Reading the files",
  fetching: "Fetching from GitHub",
  indexing: "Chunking and embedding",
};

function Progress({ job }: { job: IngestJob }) {
  // Null while the total is unknown — a GitHub job does not know how many
  // documents it has until the tree listing comes back, and a bar that
  // invents a denominator to fill is worse than one that admits it.
  const percent =
    job.total && job.total > 0
      ? Math.min(100, Math.round((job.done / job.total) * 100))
      : null;

  return (
    <section className="space-y-2">
      <h3 className="font-medium text-foreground text-xs">{job.label}</h3>
      <div className="space-y-2 rounded-xl p-3 ring-1 ring-foreground/10">
        <div className="flex items-center gap-2 text-xs">
          <IconLoader2
            className="shrink-0 animate-spin text-muted-foreground motion-reduce:animate-none"
            size={13}
          />
          <span className="text-muted-foreground">
            {PHASES[job.phase ?? ""] ?? "Working"}
          </span>
          <span className="ml-auto shrink-0 font-mono text-muted-foreground text-[11px] tabular-nums">
            {job.total !== null ? `${job.done} / ${job.total}` : job.done || ""}
          </span>
        </div>

        <div
          aria-label="Indexing progress"
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={percent ?? undefined}
          className="h-1 overflow-hidden rounded-full bg-muted"
          role="progressbar"
        >
          <span
            className={cn(
              "block h-full rounded-full bg-foreground/40",
              percent === null
                ? "w-full animate-pulse motion-reduce:animate-none"
                : "transition-[width] duration-300",
            )}
            style={percent === null ? undefined : { width: `${percent}%` }}
          />
        </div>

        {job.current && (
          <p className="truncate font-mono text-[11px] text-muted-foreground">
            {job.current}
          </p>
        )}
      </div>
    </section>
  );
}

const MARK: Record<string, string> = {
  created: "+",
  updated: "~",
  unchanged: "=",
  failed: "!",
};

function ResultRow({ result, index }: { result: IngestResult; index: number }) {
  const failed = result.status === "failed";
  return (
    <div className={cn("flex flex-col gap-0.5", index > 0 && "pt-0.5")}>
      <div className="flex gap-2">
        <span className={cn("w-3 shrink-0", failed ? "text-destructive" : "text-muted-foreground")}>
          {MARK[result.status] ?? "?"}
        </span>
        <span className={cn("min-w-0 flex-1 truncate", failed ? "text-destructive" : "text-foreground")}>
          {result.path}
        </span>
        <span className="shrink-0 text-muted-foreground tabular-nums">
          {failed ? "failed" : result.chunks ? `${result.chunks} chunks` : "no change"}
        </span>
      </div>
      {/* The reason, on the row it belongs to. One unreadable file no longer
          takes the batch down, so it has to say why it is the one that went. */}
      {failed && result.error && (
        <p className="pl-5 text-[10px] text-destructive/80 leading-4">{result.error}</p>
      )}
    </div>
  );
}

function Report({ report }: { report: IngestResponse }) {
  return (
    <section className="space-y-2">
      <h3 className="font-medium text-foreground text-xs">Result</h3>
      <div className="space-y-1 rounded-xl p-3 font-mono text-[11px] ring-1 ring-foreground/10">
        {/* Keyed by position, not by path: several files in one report can
            share a name — a repository with a README per package, say — and a
            duplicate key drops all but one row from the screen. */}
        {report.results.map((r, i) => (
          <ResultRow index={i} key={`${i}-${r.path}`} result={r} />
        ))}
        <p className="border-border border-t pt-2 text-muted-foreground">
          {report.written} written, {report.unchanged} unchanged
          {report.failed > 0 && (
            <span className="text-destructive">, {report.failed} failed</span>
          )}
          {" · "}
          <span className="tabular-nums">{report.chunks}</span> chunks · budget{" "}
          <span className="tabular-nums">{report.chunk_budget}</span> tokens
        </p>
      </div>
    </section>
  );
}

function DocumentRow({
  doc,
  onDelete,
}: {
  doc: DocumentSummary;
  onDelete?: () => void;
}) {
  const Icon = ICONS[doc.source_type];
  const body = (
    <>
      <Icon aria-hidden className="shrink-0 text-muted-foreground" size={14} stroke={1.6} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium text-foreground text-xs">
          {doc.title}
        </span>
        <span className="block truncate font-mono text-[11px] text-muted-foreground">
          {doc.path}
        </span>
      </span>
      <Badge className="shrink-0 font-normal" variant="outline">
        <span className="font-mono tabular-nums">{doc.chunk_count}</span>
      </Badge>
    </>
  );

  return (
    <li className="group relative flex items-center bg-card">
      {doc.url ? (
        <a
          className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2.5 transition-colors hover:bg-accent/60"
          href={doc.url}
          rel="noreferrer noopener"
          target="_blank"
        >
          {body}
        </a>
      ) : (
        <div className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2.5">
          {body}
        </div>
      )}
      {onDelete && (
        <button
          aria-label={`Remove ${doc.path}`}
          className="mr-2 shrink-0 rounded-sm p-1.5 text-muted-foreground opacity-0 transition-opacity hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
          onClick={onDelete}
          type="button"
        >
          <IconTrash size={13} />
        </button>
      )}
    </li>
  );
}
