/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#07080c",
          900: "#0d0f16",
          850: "#12151f",
          800: "#181c28",
          700: "#232838",
          600: "#2f3549",
          500: "#454d66",
          400: "#6b7493",
          300: "#99a1bd",
          200: "#c5cbe0",
          100: "#e6e9f3",
        },
        accent: {
          DEFAULT: "#7c9cff",
          soft: "#2a3767",
          strong: "#a9bcff",
        },
      },
      fontFamily: {
        sans: ["Inter", "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
