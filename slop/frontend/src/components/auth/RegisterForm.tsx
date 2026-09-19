import { FormEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UserPlus } from "lucide-react";
import { register } from "../../api/auth";
import { useHumanSignal } from "../../hooks/useHumanSignal";
import { strongPassword } from "../../utils/validators";
import { Button } from "../ui/Button";
import { track } from "../../utils/analytics";

export function RegisterForm() {
  const navigate = useNavigate();
  const human = useHumanSignal();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [enable2fa, setEnable2fa] = useState(false);
  const [website, setWebsite] = useState("");
  const [loading, setLoading] = useState(false);
  const strength = useMemo(() => (strongPassword(password) ? 100 : Math.min(85, password.length * 9)), [password]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human) return;
    setLoading(true);
    try {
      track("signup_started");
      await register(email, password, confirm, enable2fa, website);
      track("signup_completed", { enable2fa });
      navigate("/login");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="auth-form" onSubmit={submit}>
      <input
        name="website"
        value={website}
        onChange={(event) => setWebsite(event.target.value)}
        tabIndex={-1}
        className="honeypot"
        autoComplete="off"
        aria-hidden="true"
        aria-label="Do not fill"
      />
      <label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
      <label>Password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
      <div className="meter"><span style={{ width: `${strength}%` }} /></div>
      <label>Confirm<input type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} required /></label>
      <label className="toggle"><input type="checkbox" checked={enable2fa} onChange={(event) => setEnable2fa(event.target.checked)} /> Enable 2FA seed</label>
      <Button icon={<UserPlus size={16} />} disabled={loading || password !== confirm}>Register</Button>
    </form>
  );
}
