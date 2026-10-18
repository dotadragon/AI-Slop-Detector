import { useQuery } from "@tanstack/react-query";
import { RotateCw, ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Button } from "../components/ui/Button";
import { useScanHistory, type ScanHistoryEntry } from "../hooks/useScanHistory";
import { useSeo } from "../hooks/useSeo";

type HistoryItem = {
  job_id: string;
  type: string;
  score?: number;
  status: string;
  created_at: string;
};

function ServerJobRow({ item }: { item: HistoryItem }) {
  return (
    <article className="history-row">
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <span className="badge badge-amber" style={{ textTransform: "uppercase", fontSize: "11px" }}>
          {item.type}
        </span>
        <span style={{ fontFamily: "var(--mono)", fontSize: "12px", color: "var(--text3)" }}>
          {item.job_id.slice(0, 8)}…
        </span>
      </div>
      <span>{item.status}</span>
      <span style={{ fontWeight: 600 }}>
        {item.score == null ? "--" : `${Math.round(item.score * 100)}% AI`}
      </span>
      <time>{new Date(item.created_at).toLocaleString()}</time>
    </article>
  );
}

function LocalHistoryRow({ item }: { item: ScanHistoryEntry }) {
  return (
    <article className="history-row">
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <span className="badge badge-ai" style={{ textTransform: "uppercase", fontSize: "11px" }}>
          {item.mode}
        </span>
        <span style={{ fontSize: "13px", color: "var(--text)" }}>
          {item.title || "Forensic Scan"}
        </span>
      </div>
      <span className="badge badge-emerald">{item.label}</span>
      <span style={{ fontWeight: 600 }}>{Math.round(item.score * 100)}% AI</span>
      <time>{new Date(item.timestamp).toLocaleString()}</time>
    </article>
  );
}

function EmptyHistoryState() {
  return (
    <div className="empty-state">
      <p>No scans recorded yet.</p>
      <Link to="/detector" className="btn btn-sm btn-primary" style={{ marginTop: "12px", display: "inline-flex", alignItems: "center", gap: "6px" }}>
        <span>Launch Detector to Scan</span>
        <ExternalLink size={14} />
      </Link>
    </div>
  );
}

export function Dashboard() {
  useSeo({
    title: "Dashboard — Scan History",
    description: "Review SLOP scan history, scores, and job states.",
    path: "/dashboard",
  });

  const { history: localHistory } = useScanHistory();

  const { data: serverJobs = [], refetch, isFetching } = useQuery<HistoryItem[]>({
    queryKey: ["history"],
    queryFn: async () => {
      try {
        const res = await api.get("/jobs/history");
        return res.data as HistoryItem[];
      } catch {
        return [];
      }
    },
    retry: 1,
  });

  return (
    <main className="dashboard">
      <div className="dashboard-head">
        <div>
          <h1>Scan History</h1>
          <p className="text-secondary" style={{ fontSize: "14px", marginTop: "4px" }}>
            Audit log of all processed text, image, and video forensic analyses.
          </p>
        </div>
        <Button icon={<RotateCw size={16} className={isFetching ? "animate-spin" : ""} />} variant="ghost" onClick={() => refetch()}>
          Refresh
        </Button>
      </div>

      <div className="history">
        {serverJobs.length > 0 ? (
          serverJobs.map((item) => <ServerJobRow key={item.job_id} item={item} />)
        ) : localHistory.length > 0 ? (
          localHistory.map((item) => <LocalHistoryRow key={item.id} item={item} />)
        ) : (
          <EmptyHistoryState />
        )}
      </div>
    </main>
  );
}
