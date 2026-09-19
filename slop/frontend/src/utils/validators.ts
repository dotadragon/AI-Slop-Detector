export function strongPassword(password: string) {
  return /[A-Z]/.test(password) && /\d/.test(password) && /[^A-Za-z0-9]/.test(password) && password.length >= 8;
}

export function scoreLabel(score: number) {
  if (score >= 0.72) return "AI Likely";
  if (score >= 0.42) return "Mixed Signal";
  return "Human Likely";
}

// ponytail: fixed color references to match v2 design tokens
export function scoreColor(score: number) {
  if (score >= 0.72) return "var(--red)";
  if (score >= 0.42) return "var(--amber)";
  return "var(--green)";
}
