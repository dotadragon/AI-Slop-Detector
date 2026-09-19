import { api } from "./client";
import type { User } from "../types";

export async function login(email: string, password: string, website = "") {
  const { data } = await api.post("/auth/login", { email, password, website });
  return data as { access_token: string };
}

export async function register(email: string, password: string, confirm_password: string, enable_2fa = false, website = "") {
  const { data } = await api.post("/auth/register", { email, password, confirm_password, enable_2fa, website });
  return data;
}

export async function me() {
  const { data } = await api.get<User>("/auth/me");
  return data;
}

export async function forgotPassword(email: string, website = "") {
  const { data } = await api.post("/auth/forgot-password", { email, website });
  return data;
}

export async function resetPassword(token: string, new_password: string, website = "") {
  const { data } = await api.post("/auth/reset-password", { token, new_password, website });
  return data;
}

export async function logout() {
  await api.post("/auth/logout");
}
