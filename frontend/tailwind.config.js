/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          dark: '#0b0f19',
          light: '#f8fafc',
        },
        surface: {
          dark: '#111827',
          light: '#ffffff',
        },
        card: {
          dark: '#1e293b',
          light: '#f1f5f9',
        },
        border: {
          dark: '#334155',
          light: '#e2e8f0',
        },
        sentiment: {
          pos: {
            DEFAULT: '#10b981',
            light: '#d1fae5',
            dark: '#065f46',
          },
          neg: {
            DEFAULT: '#ef4444',
            light: '#fee2e2',
            dark: '#991b1b',
          },
          neu: {
            DEFAULT: '#f59e0b',
            light: '#fef3c7',
            dark: '#92400e',
          },
        },
        brand: {
          purple: '#8b5cf6',
          blue: '#3b82f6',
          cyan: '#06b6d4',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
