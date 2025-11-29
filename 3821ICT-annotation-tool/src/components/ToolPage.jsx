import React from "react";
import { Button } from "@/components/ui/button";

export default function ToolPage({ setPage, fileInputRef }) {
  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileChosen = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setPage("processing"); // go to “Thank You / Processing”
    }
  };

  return (
    <section className="mx-auto max-w-screen-xl flex flex-col items-center py-20 px-4 min-h-[calc(100vh-64px)]">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChosen}
      />
      <div className="grid gap-8 w-full md:grid-cols-2">
        {/* Upload card */}
        <div className="border border-border rounded-lg shadow-sm bg-card flex flex-col">
          <div className="flex-1 flex items-center justify-center bg-muted rounded-t-lg p-8">
            <svg
              width="80"
              height="80"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
              className="text-muted-foreground/40"
            >
              <rect x="4" y="4" width="16" height="16" rx="2" />
              <line x1="8" y1="8" x2="16" y2="16" />
              <line x1="16" y1="8" x2="8" y2="16" />
            </svg>
          </div>
          <div className="p-6 space-y-4">
            <h2 className="text-lg font-semibold">Upload Your Video</h2>
            <p className="text-sm text-muted-foreground">
              Egestas elit dui scelerisque ut eu purus aliquam vitae habitasse.
            </p>
            <div className="flex flex-col md:flex-row gap-4">
              <Button
                className="flex-1 cursor-pointer"
                onClick={handleUploadClick}
              >
                Upload File
              </Button>
              <Button
                variant="outline"
                className="flex-1 cursor-pointer text-muted-foreground"
                onClick={() => setPage("home")}
              >
                Go Back
              </Button>
            </div>
          </div>
        </div>

        {/* Tutorial card */}
        <div className="border border-border rounded-lg shadow-sm bg-card flex flex-col">
          <div className="flex-1 flex items-center justify-center bg-muted rounded-t-lg p-8">
            <svg
              width="80"
              height="80"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
              className="text-muted-foreground/40"
            >
              <rect x="4" y="4" width="16" height="16" rx="2" />
              <polygon points="10,8 16,12 10,16" />
            </svg>
          </div>
          <div className="p-6 space-y-4">
            <h2 className="text-lg font-semibold">How To Use Tutorial</h2>
            <p className="text-sm text-muted-foreground">
              Egestas elit dui scelerisque ut eu purus aliquam vitae habitasse.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
