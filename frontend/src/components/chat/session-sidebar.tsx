"use client";

import { IconMessagePlus, IconTrash } from "@tabler/icons-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { StoredSession } from "@/lib/session-store";
import type { CorpusStats } from "@/lib/types";

export function SessionSidebar({
  sessions,
  activeId,
  stats,
  onSelect,
  onNew,
  onDelete,
}: {
  sessions: StoredSession[];
  activeId: string;
  stats: CorpusStats | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <nav
      aria-label="Chat sessions"
      className="flex h-full w-64 shrink-0 flex-col border-border border-r bg-sidebar"
    >
      <div className="flex items-center gap-2 px-3 py-3">
        <span className="flex-1 truncate font-heading font-semibold text-sidebar-foreground text-sm tracking-tight">
          Doc Search
        </span>
        <Button
          aria-label="New chat"
          className="rounded-md"
          onClick={onNew}
          size="icon-sm"
          variant="ghost"
        >
          <IconMessagePlus size={16} />
        </Button>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-2">
        {sessions.length === 0 ? (
          <p className="px-2 py-2 text-muted-foreground text-xs">
            No conversations yet.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map((session) => (
              <li className="group relative" key={session.id}>
                <button
                  className={cn(
                    "w-full truncate rounded-md py-1.5 pr-8 pl-2 text-left text-sidebar-foreground text-xs transition-colors",
                    "hover:bg-sidebar-accent focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-1",
                    session.id === activeId && "bg-sidebar-accent font-medium",
                  )}
                  onClick={() => onSelect(session.id)}
                  type="button"
                >
                  {session.title}
                </button>
                <button
                  aria-label={`Delete ${session.title}`}
                  className="absolute top-1 right-1 rounded-sm p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
                  onClick={() => onDelete(session.id)}
                  type="button"
                >
                  <IconTrash size={13} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-sidebar-border border-t px-3 py-3">
        {stats && (
          <dl className="space-y-1 text-muted-foreground text-xs">
            <div className="flex justify-between gap-2">
              <dt>Documents</dt>
              <dd className="font-mono text-sidebar-foreground tabular-nums">
                {stats.documents}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Chunks</dt>
              <dd className="font-mono text-sidebar-foreground tabular-nums">
                {stats.chunks}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Embeddings</dt>
              <dd className="truncate font-mono text-sidebar-foreground">
                {stats.embed_model}
              </dd>
            </div>
          </dl>
        )}
      </div>
    </nav>
  );
}
