import React, { useRef } from "react";
import Navbar from "./components/Navbar";
import LandingPage from "./components/LandingPage";
import AboutPage from "./components/AboutPage";
import ToolPage from "./components/ToolPage";
import ProcessingPage from "./components/ProcessingPage";
import ReadyPage from "./components/ReadyPage";
import Dashboard from "./components/Dashboard";
import LabelStudio from "./components/LabelStudio";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router";

function AppRoutes() {
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();
  const pathname = location.pathname || "/";
  let page;
  switch (pathname) {
    case "/":
      page = "home";
      break;
    default:
      page = pathname.replace(/^\//, "");
      break;
  }

  const setPage = (nextPage) => {
    const url = nextPage === "home" ? "/" : `/${nextPage}`;
    navigate(url);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar setPage={setPage} page={page} />
      <Routes>
        <Route path="/" element={<LandingPage setPage={setPage} />} />
        <Route path="/about" element={<AboutPage setPage={setPage} />} />
        <Route
          path="/tool"
          element={<ToolPage setPage={setPage} fileInputRef={fileInputRef} />}
        />
        <Route
          path="/processing"
          element={<ProcessingPage setPage={setPage} />}
        />
        <Route path="/ready" element={<ReadyPage setPage={setPage} />} />
        <Route path="/dashboard" element={<Dashboard setPage={setPage} />} />
        <Route path="/label-studio" element={<LabelStudio />} />
        <Route path="*" element={<LandingPage setPage={setPage} />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
