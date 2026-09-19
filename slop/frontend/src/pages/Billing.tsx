import React from "react";
import { CreditCard } from "lucide-react";
import { api } from "../api/client";
import { Button } from "../components/ui/Button";
import { useSeo } from "../hooks/useSeo";
import { track } from "../utils/analytics";

export function Billing() {
  useSeo({
    title: "Billing",
    description: "Choose a SLOP plan, manage upgrades, downgrades, cancellation, invoices, and receipts.",
    path: "/billing",
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "Product",
      name: "SLOP",
      description: "AI content detection for text, image, and video.",
      brand: { "@type": "Organization", name: "SLOP" }
    }
  });

  const checkout = async (plan: string) => {
    track("checkout_started", { plan });
    const { data } = await api.post("/billing/checkout", { plan });
    if (data.url) window.location.href = data.url;
  };

  return (
    <main className="content-page">
      <h1>Billing</h1>
      <div className="pricing-grid">
        {["analyst", "team", "enterprise"].map((plan) => (
          <article key={plan} className="price-card">
            <h2>{plan}</h2>
            <p>{plan === "analyst" ? "Individual forensic review." : plan === "team" ? "Shared scan history and controls." : "B2B controls, DPA, and priority support."}</p>
            <Button icon={<CreditCard size={16} />} onClick={() => checkout(plan)}>Start checkout</Button>
          </article>
        ))}
      </div>
      <p className="hint">Upgrade, downgrade, cancellation, invoice, receipt, and failed-payment flows are handled through the billing portal when Stripe keys are configured.</p>
    </main>
  );
}
