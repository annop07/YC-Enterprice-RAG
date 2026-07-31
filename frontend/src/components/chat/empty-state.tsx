"use client";

import {
  IconBinaryTree,
  IconCube,
  IconQuote,
  IconScissors,
} from "@tabler/icons-react";
import { Button } from "@/components/ui/button";

/**
 * Each chip is a question the indexed corpus can actually answer, so the
 * first click lands on a real citation rather than on a refusal. They fill
 * the composer instead of submitting — the same behaviour as the example
 * prompts in AI-Engineer's `ai-04` block.
 */
const SUGGESTIONS = [
  {
    id: "hybrid",
    icon: IconBinaryTree,
    label: "Hybrid search",
    prompt: "How does hybrid search work here, and how much does re-ranking help?",
  },
  {
    id: "chunking",
    icon: IconScissors,
    label: "Chunking",
    prompt: "What chunk size and overlap does ingestion use, and why?",
  },
  {
    id: "deploy",
    icon: IconCube,
    label: "Run it locally",
    prompt: "How do I run the whole stack locally?",
  },
  {
    id: "citations",
    icon: IconQuote,
    label: "Citations",
    prompt: "How do you stop the model from making up citations?",
  },
];

export function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-4 py-10 text-center">
      <h1 className="text-balance font-heading font-semibold text-2xl text-foreground tracking-tight sm:text-3xl">
        Ask your documents
      </h1>
      <p className="-mt-2 max-w-md text-balance text-muted-foreground text-sm">
        Hybrid search over Markdown, PDF and GitHub sources — every answer cites
        the exact lines it came from.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
        {SUGGESTIONS.map((s) => (
          <Button
            className="gap-2 rounded-full"
            key={s.id}
            onClick={() => onPick(s.prompt)}
            size="sm"
            type="button"
            variant="outline"
          >
            <s.icon size={16} />
            {s.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
