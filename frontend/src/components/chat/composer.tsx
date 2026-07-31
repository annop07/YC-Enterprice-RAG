"use client";

import { IconArrowUp, IconPlayerStopFilled } from "@tabler/icons-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ChatStatus } from "@/hooks/use-chat";

/**
 * Same frame as the prompt box in AI-Engineer (`ai-04`): a bordered card that
 * takes the focus ring as a whole, with the controls sitting inside it rather
 * than beside it. Value is lifted so the suggestion chips can fill it.
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  status,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  status: ChatStatus;
}) {
  const busy = status !== "idle";

  function submit() {
    if (busy || !value.trim()) return;
    onSubmit();
  }

  return (
    <form
      className="rounded-xl border bg-background p-2 transition-colors duration-200 focus-within:border-ring"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      {/* Growth is left to the base component's `field-sizing-content` rather
          than a scrollHeight effect — same as the prompt box in AI-Engineer. */}
      <Textarea
        className="max-h-50 min-h-11 resize-none rounded-none border-none bg-transparent! p-0 text-sm shadow-none focus-visible:border-transparent focus-visible:ring-0"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask the corpus…"
        value={value}
      />

      <div className="flex items-center gap-2">
        <p className="text-muted-foreground text-xs">
          <kbd className="font-mono">Enter</kbd> to send ·{" "}
          <kbd className="font-mono">Shift</kbd>+
          <kbd className="font-mono">Enter</kbd> for a new line
        </p>

        <div className="ml-auto">
          {busy ? (
            <Button
              aria-label="Stop generating"
              className="rounded-md"
              onClick={onStop}
              size="icon-sm"
              type="button"
              variant="outline"
            >
              <IconPlayerStopFilled size={12} />
            </Button>
          ) : (
            <Button
              aria-label="Send message"
              className="rounded-md"
              disabled={!value.trim()}
              size="icon-sm"
              type="submit"
              variant="default"
            >
              <IconArrowUp size={16} />
            </Button>
          )}
        </div>
      </div>
    </form>
  );
}
