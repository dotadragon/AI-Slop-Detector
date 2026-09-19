export type SeoConfig = {
  title: string;
  description: string;
  path: string;
  jsonLd?: Record<string, unknown>;
};

const siteUrl = import.meta.env.VITE_SITE_URL ?? "https://slop.dev";

function upsertMeta(selector: string, attrs: Record<string, string>) {
  let element = document.head.querySelector(selector) as HTMLMetaElement | HTMLLinkElement | HTMLScriptElement | null;
  if (!element) {
    element = selector.startsWith("link") ? document.createElement("link") : document.createElement("meta");
    document.head.appendChild(element);
  }
  Object.entries(attrs).forEach(([key, value]) => element?.setAttribute(key, value));
}

export function applySeo({ title, description, path, jsonLd }: SeoConfig) {
  const url = `${siteUrl}${path}`;
  document.title = `${title} | SLOP`;
  upsertMeta('meta[name="description"]', { name: "description", content: description });
  upsertMeta('link[rel="canonical"]', { rel: "canonical", href: url });
  upsertMeta('meta[property="og:title"]', { property: "og:title", content: `${title} | SLOP` });
  upsertMeta('meta[property="og:description"]', { property: "og:description", content: description });
  upsertMeta('meta[property="og:url"]', { property: "og:url", content: url });
  upsertMeta('meta[property="og:image"]', { property: "og:image", content: `${siteUrl}/og-image.svg` });
  upsertMeta('meta[name="twitter:title"]', { name: "twitter:title", content: `${title} | SLOP` });
  upsertMeta('meta[name="twitter:description"]', { name: "twitter:description", content: description });
  upsertMeta('meta[name="twitter:image"]', { name: "twitter:image", content: `${siteUrl}/og-image.svg` });

  const id = "slop-jsonld";
  document.getElementById(id)?.remove();
  if (jsonLd) {
    const script = document.createElement("script");
    script.id = id;
    script.type = "application/ld+json";
    script.textContent = JSON.stringify(jsonLd);
    document.head.appendChild(script);
  }
}
