import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense } from "react";
import { Footer } from "./components/layout/Footer";
import { CookieConsent } from "./components/layout/CookieConsent";
import { Nav } from "./components/layout/Nav";
import { Spinner } from "./components/ui/Spinner";
import { Landing } from "./pages/Landing";
import { Detector } from "./pages/Detector";
import { Calibration } from "./pages/Calibration";

import { Billing } from "./pages/Billing";
import { Changelog } from "./pages/Changelog";
import { Dashboard } from "./pages/Dashboard";
import { ForgotPassword } from "./pages/ForgotPassword";
import { Help } from "./pages/Help";
import { Login } from "./pages/Login";
import { Privacy } from "./pages/Privacy";
import { Register } from "./pages/Register";
import { ResetPassword } from "./pages/ResetPassword";
import { Security } from "./pages/Security";
import { Status } from "./pages/Status";
import { Terms } from "./pages/Terms";

const queryClient = new QueryClient();

const ROUTE_CONFIG = [
  { path: "/", element: <Landing /> },
  { path: "/detector", element: <Detector /> },
  { path: "/calibration", element: <Calibration /> },
  { path: "/login", element: <Login /> },
  { path: "/register", element: <Register /> },
  { path: "/forgot-password", element: <ForgotPassword /> },
  { path: "/reset-password", element: <ResetPassword /> },
  { path: "/dashboard", element: <Dashboard /> },
  { path: "/billing", element: <Billing /> },
  { path: "/privacy", element: <Privacy /> },
  { path: "/terms", element: <Terms /> },
  { path: "/security", element: <Security /> },
  { path: "/status", element: <Status /> },
  { path: "/help", element: <Help /> },
  { path: "/whats-new", element: <Changelog /> },
  {
    path: "*",
    element: (
      <main className="auth-page">
        <h1>404</h1>
        <Link to="/detector">Detector</Link>
      </main>
    ),
  },
];

function AppRoutes() {
  return (
    <Routes>
      {ROUTE_CONFIG.map(({ path, element }) => (
        <Route key={path} path={path} element={element} />
      ))}
    </Routes>
  );
}

function LoadingFallback() {
  return (
    <main className="auth-page">
      <Spinner />
    </main>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Nav />
        <Suspense fallback={<LoadingFallback />}>
          <AppRoutes />
        </Suspense>
        <CookieConsent />
        <Footer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
