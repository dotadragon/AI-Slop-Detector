import React from "react";
import { Link, NavLink } from "react-router-dom";
import { LogOut, Moon, Sun } from "lucide-react";
import { Button } from "../ui/Button";
import { useAuth } from "../../hooks/useAuth";
import { useTheme } from "../../store/theme";

export function Nav() {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();

  return (
    <nav className="nav" aria-label="Main Navigation">
      <Link to="/" className="brand" aria-label="SLOP home">
        <span className="brand-amber-dot" />
        <span className="brand-text">SLOP DETECTOR</span>
      </Link>

      <div className="nav-links">
        <NavLink to="/detector" className="nav-link-item">Detector</NavLink>
        <NavLink to="/dashboard" className="nav-link-item">Dashboard</NavLink>
        <NavLink to="/calibration" className="nav-link-item">Calibration</NavLink>
        <NavLink to="/help" className="nav-link-item">Docs</NavLink>
        <button
          className="theme-toggle"
          onClick={toggle}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        {user ? (
          <Button variant="ghost" icon={<LogOut size={16} />} onClick={logout}>
            Exit
          </Button>
        ) : (
          <>
            <NavLink to="/login" className="nav-login-link">
              Login
            </NavLink>
            <NavLink to="/detector" className="btn btn-sm btn-get-started">
              Get Started
            </NavLink>
          </>
        )}
      </div>
    </nav>
  );
}
