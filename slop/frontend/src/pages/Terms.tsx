import { useSeo } from "../hooks/useSeo";

export function Terms() {
  useSeo({
    title: "Terms of Service",
    description: "SLOP terms for responsible AI-content detection and acceptable use.",
    path: "/terms"
  });
  return (
    <main className="content-page">
      <h1>Terms of Service</h1>
      <p>SLOP provides probabilistic detection signals, not legal, employment, academic, or moderation verdicts.</p>
      <section><h2>Use Of Results</h2><p>You are responsible for independent review before taking action based on a score.</p></section>
      <section><h2>Uploads</h2><p>You must have the right to submit content. Do not upload illegal content, secrets, or personal data unless your organization has approved that processing.</p></section>
      <section><h2>Subscriptions</h2><p>Paid plans renew until cancelled. Failed payments may pause access after dunning notices.</p></section>
      <section><h2>Acceptable Use</h2><p>Do not attack the service, bypass limits, reverse engineer private models, or use SLOP to harass people.</p></section>
    </main>
  );
}
