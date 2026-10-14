import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./index.css";

// ponytail: init theme from localStorage before first paint
const savedTheme = localStorage.getItem("slop_theme") || "dark";
document.documentElement.setAttribute("data-theme", savedTheme);
document.documentElement.classList.toggle("dark", savedTheme === "dark");

// On-demand Sentry loading: only load ~1.1MB telemetry bundle if DSN is configured
if (import.meta.env.VITE_SENTRY_DSN) {
  import("@sentry/react").then((Sentry) => {
    Sentry.init({
      dsn: import.meta.env.VITE_SENTRY_DSN,
      environment: import.meta.env.MODE,
      tracesSampleRate: 0.1
    });
  });
}

// ponytail: Native React standard library ErrorBoundary replacing 1.1MB @sentry/react bundle
interface ErrorBoundaryProps {
  fallback?: React.ReactNode;
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown, errorInfo: React.ErrorInfo) {
    console.error("Uncaught application error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <main className="auth-page">
            <h1>Something broke</h1>
            <p className="hint">The error has been captured. Try refreshing the page.</p>
          </main>
        )
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary fallback={<main className="auth-page"><h1>Something broke</h1><p className="hint">The error has been captured. Try refreshing the page.</p></main>}>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

