import axios, { InternalAxiosRequestConfig } from "axios";
import { getCookie } from "../utils/sanitize";
import { useAuthStore } from "../store/authStore";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api/v1",
  withCredentials: true,
});

let csrfPromise: Promise<void> | null = null;
let inMemoryCsrf: string | null = null;

async function requestCsrfToken(baseURL: string): Promise<void> {
  try {
    const res = await fetch(`${baseURL}/auth/csrf`, { credentials: "include" });
    if (res.ok) {
      const data = await res.json();
      if (data.csrf_token) inMemoryCsrf = data.csrf_token;
    }
  } catch {
    // Ignore fetch error
  } finally {
    csrfPromise = null;
  }
}

async function ensureCsrfToken(method?: string, url?: string): Promise<string | null> {
  const isMutating = method === "POST" || method === "DELETE";
  const isAuthCsrf = url?.includes("/auth/csrf");
  let csrf = getCookie("csrf_token") || inMemoryCsrf;

  if (isMutating && !csrf && !isAuthCsrf) {
    if (!csrfPromise) {
      csrfPromise = requestCsrfToken(api.defaults.baseURL ?? "/api/v1");
    }
    await csrfPromise;
    csrf = getCookie("csrf_token") || inMemoryCsrf;
  }
  return csrf;
}

api.interceptors.request.use(async (config: InternalAxiosRequestConfig) => {
  const method = config.method?.toUpperCase();
  const csrf = await ensureCsrfToken(method, config.url);
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
    const shouldRefresh = error.response?.status === 401 && !original?._retry && !isAuthEndpoint && hadToken;

    if (shouldRefresh) {
      if (isRefreshing) return Promise.reject(error);
      original._retry = true;
      isRefreshing = true;
      try {
        const { data } = await api.post("/auth/refresh");
        useAuthStore.getState().setAccessToken(data.access_token);
        original.headers.Authorization = `Bearer ${data.access_token}`;
        return api(original);
      } catch (refreshErr) {
        useAuthStore.getState().logout();
        return Promise.reject(refreshErr);
      } finally {
        isRefreshing = false;
      }
    }
    return Promise.reject(error);
  }
);
