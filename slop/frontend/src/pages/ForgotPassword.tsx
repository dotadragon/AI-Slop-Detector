import React, { FormEvent, useState } from "react";
import { Mail } from "lucide-react";
import { forgotPassword } from "../api/auth";
import { Button } from "../components/ui/Button";
import { useHumanSignal } from "../hooks/useHumanSignal";
import { useSeo } from "../hooks/useSeo";

export function ForgotPassword() {
  useSeo({ title: "Forgot Password", description: "Request a secure SLOP password reset link.", path: "/forgot-password" });
  const human = useHumanSignal();
  const [email, setEmail] = useState("");
  const [website, setWebsite] = useState("");
  const [sent, setSent] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human) return;
    await forgotPassword(email, website);
    setSent(true);
  };
  return (
    <main className="auth-page">
      <h1>Reset Access</h1>
      <form className="auth-form" onSubmit={submit}>
        <input name="website" value={website} onChange={(event) => setWebsite(event.target.value)} tabIndex={-1} className="honeypot" autoComplete="off" />
        <label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
        <Button icon={<Mail size={16} />}>Send Link</Button>
      </form>
      {sent && <p className="hint">If the email exists, a reset link has been sent.</p>}
    </main>
  );
}
