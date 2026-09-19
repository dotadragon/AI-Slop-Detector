import React from "react";
import { useSeo } from "../hooks/useSeo";

export function Privacy() {
  useSeo({
    title: "Privacy Policy",
    description: "How SLOP collects, protects, exports, and deletes user data.",
    path: "/privacy"
  });
  return (
    <main className="content-page">
      <h1>Privacy Policy</h1>
      <p>SLOP processes submitted text, images, and videos only to provide AI-content signal analysis and maintain account security.</p>
      <section><h2>Data We Collect</h2><p>Account email, authentication events, scan job metadata, uploaded samples, detection results, billing status, and consent preferences.</p></section>
      <section><h2>Analytics</h2><p>Product analytics are opt-in and used for activation, retention, funnel, and reliability decisions. SLOP does not intentionally collect analytics properties containing private scan content.</p></section>
      <section><h2>Export And Deletion</h2><p>Users may request export or deletion through support@slop.dev. B2B customers can request a Data Processing Agreement before production use.</p></section>
      <section><h2>Retention</h2><p>Security audit logs are retained for abuse prevention. Uploaded evidence can be deleted on request or by configured retention policy.</p></section>
    </main>
  );
}
