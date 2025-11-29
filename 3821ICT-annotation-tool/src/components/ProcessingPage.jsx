import React, { useEffect, useRef, useState } from "react";
import happyBot from "../assets/cybella_happy.gif";

export default function ProcessingPage({ setPage }) {
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(false);
  const navTimeoutRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    // --- Tunables ---
    const TOTAL_MS = 2500; // total simulated processing time
    const TICK_MS = 80; // update frequency
    const HOLD_MS = 3500; // pause after "done" before moving on
    // -----------------

    const increment = (100 * TICK_MS) / TOTAL_MS;

    intervalRef.current = setInterval(() => {
      setProgress((p) => {
        const next = Math.min(p + increment, 100);
        if (next >= 100) {
          clearInterval(intervalRef.current);
          setDone(true);
          navTimeoutRef.current = setTimeout(() => {
            setPage?.("ready");
          }, HOLD_MS);
        }
        return next;
      });
    }, TICK_MS);

    return () => {
      clearInterval(intervalRef.current);
      if (navTimeoutRef.current) clearTimeout(navTimeoutRef.current);
    };
  }, [setPage]);

  const pct = Math.round(Math.min(progress, 100));

  return (
    <section className="mx-auto max-w-screen-xl flex flex-col items-center justify-center gap-8 py-24 px-4 min-h-[calc(100vh-64px)]">
      {!done ? (
        <>
          <h1 className="text-2xl md:text-3xl font-semibold text-center">
            Processing your video…
          </h1>

          {/* Loading bar */}
          <div className="w-full">
            <div className="h-4 w-full rounded-full bg-muted relative overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-300 ease-linear"
                style={{ width: `${pct}%` }}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
                role="progressbar"
              />
            </div>
            <div className="mt-2 text-sm text-muted-foreground text-right tabular-nums">
              {pct}%
            </div>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center text-center">
          {/* Circular avatar mask to avoid square white edges */}
          <div
            className={`w-40 h-40 rounded-full overflow-hidden ring-2 ring-primary/20 shadow-md bg-background dark:bg-card 
                        transition-all duration-700 ease-out ${
                          done ? "opacity-100 scale-100" : "opacity-0 scale-95"
                        }`}
          >
            <img
              src={happyBot}
              alt="Processing finished"
              className="w-full h-full object-cover"
            />
          </div>
          <p className="mt-4 text-xl md:text-2xl font-semibold">
            Processing is finished
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Taking you to results…
          </p>
        </div>
      )}
    </section>
  );
}
