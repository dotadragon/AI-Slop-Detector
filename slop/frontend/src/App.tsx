import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, lazy } from "react";
import { Footer } from "./components/layout/Footer";
import { CookieConsent } from "./components/layout/CookieConsent";
import { Nav } from "./components/layout/Nav";
import { Spinner } from "./components/ui/Spinner";
import { Landing } from "./pages/Landing";
import { Detector } from "./pages/Detector";
import { Calibration } from "./pages/Calibration";

// Secondary routes lazy-loaded to prevent initial bundle bloat
const Billing = lazy(() => import("./pages/Billing").then((m) => ({ default: m.Billing })));
const Changelog = lazy(() => import("./pages/Changelog").then((m) => ({ default: m.Changelog })));
const Dashboard = lazy(() => import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword").then((m) => ({ default: m.ForgotPassword })));
const Help = lazy(() => import("./pages/Help").then((m) => ({ default: m.Help })));
const Login = lazy(() => import("./pages/Login").then((m) => ({ default: m.Login })));
const Privacy = lazy(() => import("./pages/Privacy").then((m) => ({ default: m.Privacy })));
const Register = lazy(() => import("./pages/Register").then((m) => ({ default: m.Register })));
const ResetPassword = lazy(() => import("./pages/ResetPassword").then((m) => ({ default: m.ResetPassword })));
const Security = lazy(() => import("./pages/Security").then((m) => ({ default: m.Security })));
const Status = lazy(() => import("./pages/Status").then((m) => ({ default: m.Status })));
const Terms = lazy(() => import("./pages/Terms").then((m) => ({ default: m.Terms })));

const queryClient = new QueryClient();

function LoadingFallback() {
  return (
    <main className="auth-page">
      <Spinner />
    </main>
  );
}

function withSuspense(component: React.ReactNode) {
  return <Suspense fallback={<LoadingFallback />}>{component}</Suspense>;
}

const ROUTE_CONFIG = [
  { path: "/", element: <Landing /> },
  { path: "/detector", element: <Detector /> },
  { path: "/calibration", element: <Calibration /> },
  { path: "/login", element: withSuspense(<Login />) },
  { path: "/register", element: withSuspense(<Register />) },
  { path: "/forgot-password", element: withSuspense(<ForgotPassword />) },
  { path: "/reset-password", element: withSuspense(<ResetPassword />) },
  { path: "/dashboard", element: withSuspense(<Dashboard />) },
  { path: "/billing", element: withSuspense(<Billing />) },
  { path: "/privacy", element: withSuspense(<Privacy />) },
  { path: "/terms", element: withSuspense(<Terms />) },
  { path: "/security", element: withSuspense(<Security />) },
  { path: "/status", element: withSuspense(<Status />) },
  { path: "/help", element: withSuspense(<Help />) },
  { path: "/whats-new", element: withSuspense(<Changelog />) },
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

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Nav />
        <AppRoutes />
        <CookieConsent />
        <Footer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
