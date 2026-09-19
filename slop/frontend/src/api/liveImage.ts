export function liveImageSocketUrl() {
  const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
  const url = new URL(apiBase, window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/$/, "")}/image/live`;
  url.search = "";
  return url.toString();
}
