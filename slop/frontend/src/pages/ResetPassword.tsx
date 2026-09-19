import React from "react";
import { ResetPasswordForm } from "../components/auth/ResetPasswordForm";
import { useSeo } from "../hooks/useSeo";

export function ResetPassword() {
  useSeo({ title: "Reset Password", description: "Set a new SLOP password with a short-lived reset token.", path: "/reset-password" });
  return <main className="auth-page"><h1>New Password</h1><ResetPasswordForm /></main>;
}
