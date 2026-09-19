import React from "react";
import { useSeo } from "../hooks/useSeo";

export function Changelog() {
  useSeo({ title: "What's New", description: "SLOP launch notes and product changes.", path: "/whats-new" });
  return (
    <main className="content-page">
      <h1>What's New</h1>
      <article className="history-row"><strong>Launch readiness</strong><span>Analytics, SEO, legal, billing, monitoring, and launch ops scaffolded.</span></article>
    </main>
  );
}
