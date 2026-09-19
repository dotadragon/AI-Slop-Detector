type AnalyticsEvent =
  | "page_view"
  | "cta_clicked"
  | "signup_started"
  | "signup_completed"
  | "login_completed"
  | "scan_started"
  | "scan_completed"
  | "checkout_started"
  | "subscription_changed"
  | "onboarding_completed";

const analyticsKey = import.meta.env.VITE_POSTHOG_KEY;
const analyticsHost = import.meta.env.VITE_POSTHOG_HOST ?? "https://app.posthog.com";

export const activationEvents: AnalyticsEvent[] = [
  "signup_completed",
  "scan_started",
  "scan_completed",
  "checkout_started",
  "subscription_changed",
  "onboarding_completed"
];

// Lazy singleton reference
let posthogClient: typeof import("posthog-js").default | null = null;

async function getPostHog() {
  if (!analyticsKey) return null;
  if (!posthogClient) {
    const mod = await import("posthog-js");
    posthogClient = mod.default;
  }
  return posthogClient;
}

export async function initAnalytics(consent: boolean) {
  if (!analyticsKey) return;
  const posthog = await getPostHog();
  if (!posthog) return;

  posthog.init(analyticsKey, {
    api_host: analyticsHost,
    autocapture: false,
    capture_pageview: false,
    persistence: consent ? "localStorage+cookie" : "memory"
  });
  if (consent) posthog.opt_in_capturing();
  else posthog.opt_out_capturing();
}

export async function track(event: AnalyticsEvent, properties: Record<string, unknown> = {}) {
  if (!analyticsKey) return;
  const posthog = await getPostHog();
  if (!posthog) return;

  posthog.capture(event, {
    app: "slop",
    ...properties
  });
}

