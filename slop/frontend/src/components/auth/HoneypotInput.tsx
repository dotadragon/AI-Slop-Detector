interface HoneypotInputProps {
  value: string;
  onChange: (value: string) => void;
}

export function HoneypotInput({ value, onChange }: HoneypotInputProps) {
  return (
    <input
      name="website"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      tabIndex={-1}
      className="honeypot"
      autoComplete="off"
      aria-hidden="true"
      aria-label="Do not fill"
    />
  );
}
