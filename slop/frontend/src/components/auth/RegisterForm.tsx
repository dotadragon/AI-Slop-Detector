import { FormEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UserPlus } from "lucide-react";
import { register } from "../../api/auth";
import { useHumanSignal } from "../../hooks/useHumanSignal";
import { strongPassword } from "../../utils/validators";
import { Button } from "../ui/Button";
import { track } from "../../utils/analytics";
import { HoneypotInput } from "./HoneypotInput";

const MAX_STRENGTH = 100;
const PARTIAL_MAX = 85;
const CHAR_WEIGHT = 9;

function computePasswordStrength(password: string): number {
  if (strongPassword(password)) return MAX_STRENGTH;
  return Math.min(PARTIAL_MAX, password.length * CHAR_WEIGHT);
}

function PasswordStrengthMeter({ strength }: { strength: number }) {
  return (
    <div className="meter">
      <span style={{ width: `${strength}%` }} />
    </div>
  );
}

export function RegisterForm() {
  const navigate = useNavigate();
  const human = useHumanSignal();
  const [form, setForm] = useState({
    email: "",
    password: "",
    confirm: "",
    enable2fa: false,
    website: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const strength = useMemo(() => computePasswordStrength(form.password), [form.password]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!human) return;
    setIsSubmitting(true);
    try {
      track("signup_started");
      await register(form.email, form.password, form.confirm, form.enable2fa, form.website);
      track("signup_completed", { enable2fa: form.enable2fa });
      navigate("/login");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form className="auth-form" onSubmit={submit}>
      <HoneypotInput value={form.website} onChange={(website) => setForm((prev) => ({ ...prev, website }))} />
      <label>
        Email
        <input type="email" value={form.email} onChange={(e) => setForm((prev) => ({ ...prev, email: e.target.value }))} required />
      </label>
      <label>
        Password
        <input type="password" value={form.password} onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))} required />
      </label>
      <PasswordStrengthMeter strength={strength} />
      <label>
        Confirm
        <input type="password" value={form.confirm} onChange={(e) => setForm((prev) => ({ ...prev, confirm: e.target.value }))} required />
      </label>
      <label className="toggle">
        <input type="checkbox" checked={form.enable2fa} onChange={(e) => setForm((prev) => ({ ...prev, enable2fa: e.target.checked }))} />
        Enable 2FA seed
      </label>
      <Button icon={<UserPlus size={16} />} disabled={isSubmitting || form.password !== form.confirm}>
        Register
      </Button>
    </form>
  );
}
