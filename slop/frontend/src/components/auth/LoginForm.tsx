import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { KeyRound } from "lucide-react";
import { useHumanSignal } from "../../hooks/useHumanSignal";
import { useAuth } from "../../hooks/useAuth";
import { Button } from "../ui/Button";
import { track } from "../../utils/analytics";
import { HoneypotInput } from "./HoneypotInput";

export function LoginForm() {
  const navigate = useNavigate();
  const human = useHumanSignal();
  const { loginMutation } = useAuth();
  const [credentials, setCredentials] = useState({
    email: "",
    password: "",
    website: "",
  });

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human) return;
    await loginMutation.mutateAsync(credentials);
    track("login_completed");
    navigate("/detector");
  };

  return (
    <form className="auth-form" onSubmit={submit}>
      <HoneypotInput
        value={credentials.website}
        onChange={(website) => setCredentials((prev) => ({ ...prev, website }))}
      />
      <label>
        Email
        <input
          type="email"
          value={credentials.email}
          onChange={(event) =>
            setCredentials((prev) => ({ ...prev, email: event.target.value }))
          }
          required
        />
      </label>
      <label>
        Password
        <input
          type="password"
          value={credentials.password}
          onChange={(event) =>
            setCredentials((prev) => ({ ...prev, password: event.target.value }))
          }
          required
        />
      </label>
      <Button icon={<KeyRound size={16} />} disabled={loginMutation.isPending}>
        Login
      </Button>
    </form>
  );
}
