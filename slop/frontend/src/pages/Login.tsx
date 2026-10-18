import { Link } from "react-router-dom";
import { LoginForm } from "../components/auth/LoginForm";
import { useSeo } from "../hooks/useSeo";

export function Login() {
  useSeo({ title: "Login", description: "Log in to SLOP to review scan history and billing.", path: "/login" });
  return <main className="auth-page"><h1>Login</h1><LoginForm /><Link to="/forgot-password">Forgot password?</Link></main>;
}
