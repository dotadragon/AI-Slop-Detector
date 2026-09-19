import React from "react";
import { useSeo } from "../hooks/useSeo";

export function Security() {
  useSeo({
    title: "Security",
    description: "SLOP security posture, responsible disclosure, and production controls.",
    path: "/security"
  });
  return (
    <main className="content-page">
      <h1>Security</h1>
      <p>Report vulnerabilities to security@slop.dev. Please include impact, reproduction steps, and affected endpoints.</p>
      <section><h2>Controls</h2><p>RS256 JWTs, httpOnly refresh cookies, CSRF protection, rate limits, strict headers, upload validation, structured audit logs, and private file storage.</p></section>
      <section><h2>Disclosure</h2><p>We acknowledge reports within five business days and prioritize fixes by exploitability and user impact.</p></section>
    </main>
  );
}
