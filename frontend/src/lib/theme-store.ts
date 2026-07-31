/**
 * The active theme, read from the DOM.
 *
 * The class on <html> is the source of truth — an inline script in
 * `layout.tsx` sets it before first paint, so React state seeded from
 * localStorage could disagree with what is already on screen. Exposing it as
 * an external store lets the toggle read the real value without an effect
 * that writes state on mount.
 */
const KEY = "enterprise-rag:theme";

let listeners: (() => void)[] = [];

export function subscribeTheme(callback: () => void): () => void {
  listeners = [...listeners, callback];
  return () => {
    listeners = listeners.filter((l) => l !== callback);
  };
}

export function getThemeSnapshot(): boolean {
  return document.documentElement.classList.contains("dark");
}

export function getThemeServerSnapshot(): boolean {
  return false;
}

export function setDarkTheme(dark: boolean): void {
  document.documentElement.classList.toggle("dark", dark);
  try {
    localStorage.setItem(KEY, dark ? "dark" : "light");
  } catch {
    // Private mode — the class still applies for this page load.
  }
  for (const l of listeners) l();
}
