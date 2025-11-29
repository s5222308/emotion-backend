import React from "react";
import { Button } from "@/components/ui/button";

export default function ReadyPage({ setPage }) {
  const downloadCsv = () => {
    const rows = [
      "timestamp,label",
      "00:00:01,Start",
      "00:00:07,Action",
      "00:00:12,End",
    ].join("\n");
    const blob = new Blob([rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="mx-auto max-w-screen-xl py-24 px-4 min-h-[calc(100vh-64px)]">
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] items-center gap-6">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold">
            Your file is ready!
          </h1>
          <p className="mt-3 text-muted-foreground">
            Rhoncus morbi et augue nec, in id ullamcorper at sit.
          </p>
        </div>
        <Button
          variant="outline"
          className="h-12 px-6 cursor-pointer"
          onClick={() => setPage("home")}
        >
          Home Page
        </Button>
        <Button className="h-12 px-6 cursor-pointer" onClick={downloadCsv}>
          Download CSV
        </Button>
      </div>
    </section>
  );
}
