import { FormEvent, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { RotateCcw } from "lucide-react";
import { resetPassword } from "../../api/auth";
import { useHumanSignal } from "../../hooks/useHumanSignal";
import { Button } from "../ui/Button";

export function ResetPasswordForm() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const human = useHumanSignal();
  const token = params.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [website, setWebsite] = useState("");
  const [seconds, setSeconds] = useState(15 * 60);

  useEffect(() => {
    const id = window.setInterval(() => setSeconds((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearInterval(id);
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human || password !== confirm) return;
    await resetPassword(token, password, website);
    navigate("/login");
  };

  return (
    <form className="auth-form" onSubmit={submit}>
      <input name="website" value={website} onChange={(event) => setWebsite(event.target.value)} tabIndex={-1} className="honeypot" autoComplete="off" />
      <small>Expires in {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}</small>
      <label>New password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
      <label>Confirm<input type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} required /></label>
      <Button icon={<RotateCcw size={16} />} disabled={!token || password !== confirm}>Reset</Button>
    </form>
  );
}
