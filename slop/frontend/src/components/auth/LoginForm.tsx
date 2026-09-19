import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { KeyRound } from "lucide-react";
import { useHumanSignal } from "../../hooks/useHumanSignal";
import { useAuth } from "../../hooks/useAuth";
import { Button } from "../ui/Button";
import { track } from "../../utils/analytics";

export function LoginForm() {
  const navigate = useNavigate();
  const human = useHumanSignal();
  const { loginMutation } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [website, setWebsite] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human) return;
    await loginMutation.mutateAsync({ email, password, website });
    track("login_completed");
    navigate("/detector");
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
      <Button icon={<KeyRound size={16} />} disabled={loginMutation.isPending}>Login</Button>
    </form>
  );
}
