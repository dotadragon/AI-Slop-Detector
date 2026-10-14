import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense, useEffect } from "react";
import Lenis from "lenis";
import { Footer } from "./components/layout/Footer";
import { CookieConsent } from "./components/layout/CookieConsent";
import { Nav } from "./components/layout/Nav";
import { Spinner } from "./components/ui/Spinner";

const Billing = lazy(() => import("./pages/Billing").then((module) => ({ default: module.Billing })));
const Calibration = lazy(() => import("./pages/Calibration").then((module) => ({ default: module.Calibration })));
const Changelog = lazy(() => import("./pages/Changelog").then((module) => ({ default: module.Changelog })));
const Dashboard = lazy(() => import("./pages/Dashboard").then((module) => ({ default: module.Dashboard })));
const Detector = lazy(() => import("./pages/Detector").then((module) => ({ default: module.Detector })));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword").then((module) => ({ default: module.ForgotPassword })));
const Help = lazy(() => import("./pages/Help").then((module) => ({ default: module.Help })));
const Landing = lazy(() => import("./pages/Landing").then((module) => ({ default: module.Landing })));
const Login = lazy(() => import("./pages/Login").then((module) => ({ default: module.Login })));
const Privacy = lazy(() => import("./pages/Privacy").then((module) => ({ default: module.Privacy })));
const Register = lazy(() => import("./pages/Register").then((module) => ({ default: module.Register })));
const ResetPassword = lazy(() => import("./pages/ResetPassword").then((module) => ({ default: module.ResetPassword })));
const Security = lazy(() => import("./pages/Security").then((module) => ({ default: module.Security })));
const Status = lazy(() => import("./pages/Status").then((module) => ({ default: module.Status })));
const Terms = lazy(() => import("./pages/Terms").then((module) => ({ default: module.Terms })));

const queryClient = new QueryClient();

export function App() {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    const lenis = new Lenis({
      duration: 2.4,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      wheelMultiplier: 0.82,
      touchMultiplier: 1.6,
    });

    (window as unknown as { __lenis?: Lenis }).__lenis = lenis;

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    const animId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(animId);
      lenis.destroy();
      delete (window as unknown as { __lenis?: Lenis }).__lenis;
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Nav />
        <Suspense fallback={<main className="auth-page"><Spinner /></main>}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/detector" element={<Detector />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPassword />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/billing" element={<Billing />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/security" element={<Security />} />
            <Route path="/status" element={<Status />} />
            <Route path="/help" element={<Help />} />
            <Route path="/whats-new" element={<Changelog />} />
            <Route path="*" element={<main className="auth-page"><h1>404</h1><Link to="/detector">Detector</Link></main>} />
          </Routes>
        </Suspense>
        <CookieConsent />
        <Footer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
