import axios from "axios";
import { getCookie } from "../utils/sanitize";
import { useAuthStore } from "../store/authStore";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api/v1",
  withCredentials: true
});

let csrfPromise: Promise<void> | null = null;
let inMemoryCsrf: string | null = null;

api.interceptors.request.use(async (config) => {
  const method = config.method?.toUpperCase();
  let csrf = getCookie("csrf_token") || inMemoryCsrf;

  // Fetch CSRF once if missing on state-mutating requests
  if ((method === "POST" || method === "DELETE") && !csrf && !config.url?.includes("/auth/csrf")) {
    if (!csrfPromise) {
      csrfPromise = fetch(`${api.defaults.baseURL}/auth/csrf`, { credentials: "include" })
        .then(async (res) => {
          if (res.ok) {
            try {
              const data = await res.json();
              if (data.csrf_token) inMemoryCsrf = data.csrf_token;
            } catch {
              // ignore parse error
            }
          }
          csrfPromise = null;
        })
        .catch(() => { csrfPromise = null; });
    }
    await csrfPromise;
    csrf = getCookie("csrf_token") || inMemoryCsrf;
  }

  const token = useAuthStore.getState().accessToken;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  if (csrf) config.headers["X-CSRF-Token"] = csrf;
  return config;
});

let isRefreshing = false;

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    const hadToken = Boolean(useAuthStore.getState().accessToken);
    const isAuthEndpoint = original?.url?.includes("/auth/");

    // Only attempt token refresh if user was logged in and not already in an auth flow
    if (error.response?.status === 401 && !original?._retry && !isAuthEndpoint && hadToken) {
      if (isRefreshing) {
        return Promise.reject(error);
      }
      original._retry = true;
      isRefreshing = true;
      try {
        const { data } = await axios.post(`${api.defaults.baseURL}/auth/refresh`, {}, { withCredentials: true });
        useAuthStore.getState().setAccessToken(data.access_token);
        original.headers.Authorization = `Bearer ${data.access_token}`;
        isRefreshing = false;
        return api(original);
      } catch {
        isRefreshing = false;
        useAuthStore.getState().logout();
      }
    }
    return Promise.reject(error);
  }
);
