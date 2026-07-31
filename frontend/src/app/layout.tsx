import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Doc Search — Enterprise RAG",
  description:
    "Ask your internal documents. Hybrid search over pgvector with line-exact citations.",
};

/**
 * Applies the stored theme before first paint.
 *
 * This is a bare <script> in <head> rather than `next/script`, which was
 * tried first: `beforeInteractive` defers inline scripts into Next's own
 * bootstrap queue (`self.__next_s`), so they run after the framework loads —
 * which is exactly the frame of wrong background this is here to prevent.
 * The cost is a React dev-mode warning about script tags inside components;
 * it is stripped from production builds.
 */
const THEME_SCRIPT = `try{var t=localStorage.getItem("enterprise-rag:theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme: dark)").matches))document.documentElement.classList.add("dark")}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      lang="en"
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="h-full">{children}</body>
    </html>
  );
}
