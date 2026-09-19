import type { ButtonHTMLAttributes, ReactNode } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode;
  variant?: "primary" | "ghost" | "danger";
};

export function Button({ icon, children, variant = "primary", className = "", ...props }: Props) {
  return (
    <button className={`btn btn-${variant} ${className}`} {...props}>
      {icon}
      <span>{children}</span>
    </button>
  );
}
