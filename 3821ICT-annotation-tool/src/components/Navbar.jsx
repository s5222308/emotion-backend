import React from "react";
import ThemeSwitcher from "./ThemeSwitcher";
import logo from "../assets/thumbnail_image.png";
import ToggleSidebarButton from "./ui/ToggleSidebarButton";

export default function Navbar({ setPage, page }) {
  return (
    <header className="w-full border-b border-border bg-background">
      <div className="flex w-full gap-4 items-cente p-4">
        <button
          className="flex items-center gap-2 focus:outline-none cursor-pointer"
          onClick={() => setPage("home")}
        >
          <img src={logo} alt="Logo" className="h-10 w-auto" />
          <div className="flex flex-col items-start leading-tight">
            <span className="text-xl font-semibold text-foreground">
              Annotation Tool
            </span>
          </div>
        </button>
        <div className="flex-1 flex justify-end">
          <nav className="flex w-full items-center justify-between gap-6 text-sm font-medium text-foreground/80">
            <div className="flex items-center bg-accent/40 rounded p-3 px-5 gap-6 text-sm font-medium text-foreground/80">
            <a onClick={()=>setPage("dashboard")} className="hover:text-foreground cursor-pointer">
              Dashboard
            </a>
            <a onClick={()=>setPage("label-studio")} className="hover:text-foreground cursor-pointer">Label Studio</a>
            <a onClick={()=>setPage("about")} className="hover:text-foreground cursor-pointer">
              About Us
            </a>
            </div>
            <nav className="flex items-center gap-4">
              {page === "label-studio" && <ToggleSidebarButton/>}
              <ThemeSwitcher />
            </nav>
          </nav>
        </div>
      </div>
    </header>
  );
}
