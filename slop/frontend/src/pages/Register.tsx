import React from "react";
import { RegisterForm } from "../components/auth/RegisterForm";
import { useSeo } from "../hooks/useSeo";

export function Register() {
  useSeo({ title: "Register", description: "Create a SLOP account and start analyzing AI-content signals.", path: "/register" });
  return <main className="auth-page"><h1>Register</h1><RegisterForm /></main>;
}
