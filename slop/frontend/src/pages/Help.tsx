import { useSeo } from "../hooks/useSeo";

export function Help() {
  useSeo({
    title: "Help Center",
    description: "Minimal SLOP help center for onboarding, scan interpretation, billing, and support.",
    path: "/help",
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: [
        { "@type": "Question", name: "Are SLOP scores verdicts?", acceptedAnswer: { "@type": "Answer", text: "No. Scores are signals for review and should be paired with human judgment." } },
        { "@type": "Question", name: "Can I delete uploaded files?", acceptedAnswer: { "@type": "Answer", text: "Yes. Contact support or use the account deletion/export flow when enabled." } }
      ]
    }
  });
  return (
    <main className="content-page">
      <h1>Help Center</h1>
      <section><h2>Getting Started</h2><p>Open Detector, choose Text, Image, or Video, submit a sample, and review the score with supporting signals.</p></section>
      <section><h2>Support</h2><p>Email support@slop.dev for account, billing, privacy, or scan-history questions.</p></section>
      <section><h2>Interpreting Results</h2><p>High scores indicate stronger AI-generation signals. Mixed results require additional review and context.</p></section>
    </main>
  );
}
