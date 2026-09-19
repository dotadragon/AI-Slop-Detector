import React from "react";
import { useSeo } from "../hooks/useSeo";

export function Status() {
  useSeo({ title: "Status", description: "SLOP service status and incident communication.", path: "/status" });
  return (
    <main className="content-page">
      <h1>Status</h1>
      <div className="status-ok">All systems operational</div>
      <p className="hint">Production uptime monitoring should point BetterStack or UptimeRobot at /health and publish incidents here.</p>
    </main>
  );
}
