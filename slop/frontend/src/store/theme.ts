import { create } from "zustand";

// ponytail: theme store — dark/light toggle persisted to localStorage
type Theme = "dark" | "light";

interface ThemeStore {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
}

export const useTheme = create<ThemeStore>((set) => ({
  theme: (localStorage.getItem("slop_theme") as Theme) || "dark",
  toggle: () =>
    set((s) => {
      const next = s.theme === "dark" ? "light" : "dark";
      localStorage.setItem("slop_theme", next);
      document.documentElement.setAttribute("data-theme", next);
      document.documentElement.classList.toggle("dark", next === "dark");
      return { theme: next };
    }),
  set: (t: Theme) => {
    localStorage.setItem("slop_theme", t);
    document.documentElement.setAttribute("data-theme", t);
    document.documentElement.classList.toggle("dark", t === "dark");
    return { theme: t };
  },
}));
