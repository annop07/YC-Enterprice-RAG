"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { IconMenu2, IconX } from "@tabler/icons-react";
import { Composer } from "@/components/chat/composer";
import { EmptyState } from "@/components/chat/empty-state";
import { MessageList } from "@/components/chat/message-list";
import { SessionSidebar } from "@/components/chat/session-sidebar";
import { SourcePanel } from "@/components/chat/source-panel";
import { ThemeToggle } from "@/components/chat/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useChat } from "@/hooks/use-chat";
import { IS_DEMO, getCorpusStats } from "@/lib/api";
import {
  deleteSession,
  deriveTitle,
  getSession,
  getSessionsServerSnapshot,
  getSessionsSnapshot,
  newSessionId,
  subscribeSessions,
  upsertSession,
} from "@/lib/session-store";
import { cn } from "@/lib/utils";
import type { CorpusStats, Source } from "@/lib/types";

export function ChatShell() {
  const sessions = useSyncExternalStore(
    subscribeSessions,
    getSessionsSnapshot,
    getSessionsServerSnapshot,
  );

  // Lazily generated once per mount. It never reaches the DOM as text — only
  // as the "which row is highlighted" comparison, and the stored session list
  // is empty during SSR — so a server/client mismatch has nothing to render.
  const [sessionId, setSessionId] = useState(newSessionId);

  const { messages, status, send, stop, load } = useChat(sessionId);
  const [draft, setDraft] = useState("");
  const [openSource, setOpenSource] = useState<Source | null>(null);
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Only a turn the user actually sent is worth persisting; without this,
  // opening an old session would immediately rewrite it and reorder the list.
  const dirty = useRef(false);

  useEffect(() => {
    let cancelled = false;
    getCorpusStats()
      .then((s) => {
        if (!cancelled) setStats(s);
      })
      .catch(() => {
        /* sidebar just omits the counts */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (status !== "idle" || !dirty.current || messages.length === 0) return;
    dirty.current = false;
    const firstUser = messages.find((m) => m.role === "user");
    upsertSession({
      id: sessionId,
      title: deriveTitle(firstUser?.content ?? "New chat"),
      updated_at: new Date().toISOString(),
      messages,
    });
  }, [status, messages, sessionId]);

  const submit = useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    dirty.current = true;
    void send(text);
  }, [draft, send]);

  const startNew = useCallback(() => {
    setSessionId(newSessionId());
    load([]);
    setDraft("");
    setOpenSource(null);
    setSidebarOpen(false);
  }, [load]);

  const openSession = useCallback(
    (id: string) => {
      const session = getSession(id);
      if (!session) return;
      setSessionId(id);
      load(session.messages);
      setOpenSource(null);
      setSidebarOpen(false);
    },
    [load],
  );

  const removeSession = useCallback(
    (id: string) => {
      deleteSession(id);
      if (id === sessionId) startNew();
    },
    [sessionId, startNew],
  );

  // --- stick-to-bottom ---------------------------------------------------
  // Tokens arrive continuously, so the transcript follows the newest text —
  // but only while the reader is already at the bottom. Scrolling up to read
  // an earlier citation must not get yanked back down.
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const activeTitle =
    messages.find((m) => m.role === "user")?.content ?? "New chat";

  return (
    <div className="flex h-full">
      {/* Sidebar: permanent from md up, an overlay below it */}
      <div className="hidden md:flex">
        <SessionSidebar
          activeId={sessionId}
          onDelete={removeSession}
          onNew={startNew}
          onSelect={openSession}
          sessions={sessions}
          stats={stats}
        />
      </div>

      {sidebarOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <SessionSidebar
            activeId={sessionId}
            onDelete={removeSession}
            onNew={startNew}
            onSelect={openSession}
            sessions={sessions}
            stats={stats}
          />
          <button
            aria-label="Close menu"
            className="flex-1 bg-foreground/10 backdrop-blur-[1px]"
            onClick={() => setSidebarOpen(false)}
            type="button"
          />
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-border border-b px-4 py-3 md:px-6">
          <Button
            aria-label={sidebarOpen ? "Close menu" : "Open menu"}
            className="rounded-md md:hidden"
            onClick={() => setSidebarOpen((v) => !v)}
            size="icon-sm"
            variant="ghost"
          >
            {sidebarOpen ? <IconX size={16} /> : <IconMenu2 size={16} />}
          </Button>

          <h1 className="min-w-0 flex-1 truncate font-heading font-medium text-foreground text-sm tracking-tight">
            {activeTitle}
          </h1>

          {IS_DEMO && (
            // Said plainly rather than hidden: nothing behind this UI is a
            // real index yet, and a screenshot should not imply otherwise.
            <Badge className="font-normal" variant="outline">
              <span className="text-muted-foreground">demo corpus</span>
            </Badge>
          )}
          <ThemeToggle />
        </header>

        <div
          className={cn(
            "scroll-thin flex-1 overflow-y-auto px-4 py-6 md:px-6",
            messages.length === 0 && "flex items-center justify-center",
          )}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
          ref={scrollRef}
        >
          <div className="mx-auto w-full max-w-3xl">
            {messages.length === 0 ? (
              <EmptyState onPick={setDraft} />
            ) : (
              <MessageList
                activeChunkId={openSource?.chunk_id ?? null}
                messages={messages}
                onOpenSource={setOpenSource}
                status={status}
              />
            )}
          </div>
        </div>

        <div className="border-border border-t px-4 py-3 md:px-6">
          <div className="mx-auto w-full max-w-3xl">
            <Composer
              onChange={setDraft}
              onStop={stop}
              onSubmit={submit}
              status={status}
              value={draft}
            />
          </div>
        </div>
      </div>

      <SourcePanel onClose={() => setOpenSource(null)} source={openSource} />
    </div>
  );
}
