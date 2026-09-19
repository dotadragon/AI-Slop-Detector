import { useEffect, useState } from "react";
import { initAnalytics, track } from "../../utils/analytics";
import { Button } from "../ui/Button";

const key = "slop_cookie_consent";

export function CookieConsent() {
  const [choice, setChoice] = useState<string | null>(() => localStorage.getItem(key));

  useEffect(() => {
    initAnalytics(choice === "accepted");
  }, [choice]);

  if (choice) return null;

  const decide = (value: "accepted" | "declined") => {
    localStorage.setItem(key, value);
    setChoice(value);
    if (value === "accepted") track("cta_clicked", { section: "cookie_banner", action: "accept" });
  };

  return (
    <div className="cookie-banner" role="dialog" aria-label="Cookie consent">
      <p>SLOP uses privacy-aware product analytics to understand activation, retention, and broken flows.</p>
      <div>
        <Button variant="ghost" onClick={() => decide("declined")}>Decline</Button>
        <Button onClick={() => decide("accepted")}>Allow analytics</Button>
      </div>
    </div>
  );
}
