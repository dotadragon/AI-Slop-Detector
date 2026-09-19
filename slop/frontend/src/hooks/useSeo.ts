import { useEffect } from "react";
import { applySeo, type SeoConfig } from "../utils/seo";
import { track } from "../utils/analytics";

export function useSeo(config: SeoConfig) {
  useEffect(() => {
    applySeo(config);
    track("page_view", { path: config.path, title: config.title });
  }, [config.title, config.description, config.path]);
}
