"use client";

import { useSyncExternalStore } from "react";
import { IconMoon, IconSun } from "@tabler/icons-react";
import { Button } from "@/components/ui/button";
import {
  getThemeServerSnapshot,
  getThemeSnapshot,
  setDarkTheme,
  subscribeTheme,
} from "@/lib/theme-store";

/**
 * Lives in the header rather than the sidebar footer: the bottom-left corner
 * is where the dev-mode Next.js indicator sits, and a control you cannot
 * click while developing is worse than one placed a little unconventionally.
 */
export function ThemeToggle() {
  const dark = useSyncExternalStore(
    subscribeTheme,
    getThemeSnapshot,
    getThemeServerSnapshot,
  );

  return (
    <Button
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="rounded-md"
      onClick={() => setDarkTheme(!dark)}
      size="icon-sm"
      variant="ghost"
    >
      {dark ? <IconSun size={16} /> : <IconMoon size={16} />}
    </Button>
  );
}
