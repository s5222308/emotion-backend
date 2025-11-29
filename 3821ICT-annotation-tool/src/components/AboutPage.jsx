import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

export default function AboutPage({ setPage }) {
  const points = [
    "AI companion for youth mental health that’s accessible ‘anytime, anywhere’.",
    "Designed to support vulnerable and isolated young people, with a particular focus on Asian communities.",
    "Centred on early support for wellbeing and suicide-prevention contexts for young people.",
    "Provides a friendly, always-available first touchpoint to reduce isolation and encourage help-seeking.",
  ];

  const [reveal, setReveal] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setReveal(true), 50);
    return () => clearTimeout(t);
  }, []);

  const CARD_STAGGER_MS = 850;
  const headingDelayMs = 0;
  const buttonDelayMs = points.length * CARD_STAGGER_MS + 500;

  return (
    <section className="mx-auto max-w-screen-xl py-16 px-4 min-h-[calc(100vh-64px)]">
      <div className="text-center">
        <span className="uppercase text-sm font-semibold text-blue-600 mb-2 inline-block">
          About Us
        </span>

        <h2
          className="text-3xl md:text-5xl font-bold mx-auto max-w-4xl text-foreground"
          style={{
            opacity: reveal ? 1 : 0,
            transform: reveal ? "none" : "translateY(8px)",
            transition:
              "opacity 600ms ease, transform 600ms cubic-bezier(.2,.8,.2,1)",
            transitionDelay: `${headingDelayMs}ms`,
          }}
        >
          A Friend Anytime, Anywhere.
        </h2>
      </div>

      <div className="mt-12 grid gap-6 md:grid-cols-2">
        {points.map((text, i) => (
          <div
            key={i}
            className="
              rounded-2xl border border-border
              bg-card text-card-foreground
              shadow-sm p-6 md:p-7
              hover:shadow-md transition-shadow
              backdrop-blur-sm
            "
            style={{
              opacity: reveal ? 1 : 0,
              transform: reveal ? "none" : "translateY(12px)",
              transition:
                "opacity 600ms ease, transform 600ms cubic-bezier(.2,.8,.2,1)",
              transitionDelay: `${(i + 1) * CARD_STAGGER_MS}ms`,
            }}
          >
            <p className="leading-relaxed text-base md:text-lg">{text}</p>
          </div>
        ))}
      </div>

      <div
        className="mt-14 flex justify-center"
        style={{
          opacity: reveal ? 1 : 0,
          transform: reveal ? "none" : "translateY(8px)",
          transition:
            "opacity 800ms ease, transform 800ms cubic-bezier(.2,.8,.2,1)",
          transitionDelay: `${buttonDelayMs}ms`,
        }}
      >
        <Button
          className="px-6 py-6 cursor-pointer"
          onClick={() => setPage("label-studio")}
        >
          Use The Tool
        </Button>
      </div>
    </section>
  );
}
