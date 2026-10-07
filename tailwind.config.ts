import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#090D14",
        surface: {
          DEFAULT: "#0F1623",
          subtle: "#141C2B",
          card: "#172033",
          hover: "#1D273E",
        },
        border: {
          DEFAULT: "#1E293B",
          subtle: "#28354D",
          hover: "#3B4A68",
        },
        accent: {
          DEFAULT: "#3B82F6",
          hover: "#2563EB",
          subtle: "rgba(59, 130, 246, 0.12)",
        },
        sentiment: {
          positive: {
            DEFAULT: "#10B981",
            bg: "rgba(16, 185, 129, 0.12)",
            border: "rgba(16, 185, 129, 0.28)",
            text: "#34D399",
          },
          negative: {
            DEFAULT: "#EF4444",
            bg: "rgba(239, 68, 68, 0.12)",
            border: "rgba(239, 68, 68, 0.28)",
            text: "#F87171",
          },
          neutral: {
            DEFAULT: "#F59E0B",
            bg: "rgba(245, 158, 11, 0.12)",
            border: "rgba(245, 158, 11, 0.28)",
            text: "#FBBF24",
          },
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
