import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sentiment AI • NLP Sentiment Analyzer",
  description:
    "Analyze the emotional tone and sentiment of any sentence, review, or message using natural language processing with instant confidence scoring.",
  keywords: [
    "NLP",
    "Sentiment Analysis",
    "Text Classification",
    "Machine Learning",
    "Sentiment AI",
  ],
  authors: [{ name: "Sentiment AI" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="min-h-screen bg-background text-slate-100 antialiased selection:bg-blue-500/20 selection:text-blue-200">
        {children}
      </body>
    </html>
  );
}
