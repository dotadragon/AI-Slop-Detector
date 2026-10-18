import { create } from "zustand";

type Theme = "dark" | "light";

interface ThemeStore {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
}

const initialTheme: Theme = (typeof localStorage !== "undefined" && (localStorage.getItem("slop_theme") as Theme)) || "dark";
if (typeof document !== "undefined") {
  document.documentElement.setAttribute("data-theme", initialTheme);
  document.documentElement.classList.toggle("dark", initialTheme === "dark");
  document.documentElement.classList.toggle("light", initialTheme === "light");
}

const useTheme = create<ThemeStore>((set) => ({
  theme: initialTheme,
  toggle: () =>
    set((s) => {
      const next = s.theme === "dark" ? "light" : "dark";
      localStorage.setItem("slop_theme", next);
      if (typeof document !== "undefined") {
        document.documentElement.setAttribute("data-theme", next);
        document.documentElement.classList.toggle("dark", next === "dark");
        document.documentElement.classList.toggle("light", next === "light");
      }
      return { theme: next };
    }),
  set: (t: Theme) => {
    localStorage.setItem("slop_theme", t);
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", t);
      document.documentElement.classList.toggle("dark", t === "dark");
      document.documentElement.classList.toggle("light", t === "light");
    }
    return { theme: t };
  },
}));

export { useTheme };
