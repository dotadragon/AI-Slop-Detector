import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { login, me, logout } from "../api/auth";
import { useAuthStore } from "../store/authStore";

export function useAuth() {
  const { accessToken, user, setAccessToken, setUser, logout: clearAuth } = useAuthStore();
  const meQuery = useQuery({
    queryKey: ["me", accessToken],
    queryFn: me,
    enabled: Boolean(accessToken),
  });

  const loginMutation = useMutation({
    mutationFn: ({ email, password, website }: { email: string; password: string; website?: string }) =>
      login(email, password, website),
    onSuccess: (data) => setAccessToken(data.access_token),
  });

  const handleLogout = async () => {
    try {
      await logout();
    } finally {
      clearAuth();
    }
  };

  useEffect(() => {
    if (meQuery.data) setUser(meQuery.data);
  }, [meQuery.data, setUser]);

  return { accessToken, user, loginMutation, logout: handleLogout };
}
