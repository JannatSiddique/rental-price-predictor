"use client";

import { useState } from "react";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const scrollTo = (id: string) => {
    setOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav className="sticky top-0 z-50 border-b border-white/10 bg-black/70 backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <button
          onClick={() => scrollTo("hero")}
          className="text-xl font-bold bg-gradient-to-r from-purple-400 to-white bg-clip-text text-transparent"
        >
          RentPredict
        </button>

        <div className="hidden sm:flex items-center gap-8 text-sm text-gray-400">
          <button onClick={() => scrollTo("hero")} className="hover:text-white transition">
            Home
          </button>
          <button onClick={() => scrollTo("predict")} className="hover:text-white transition">
            Predict
          </button>
        </div>

        <button
          className="sm:hidden text-gray-400 hover:text-white"
          onClick={() => setOpen(!open)}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {open ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>

      {open && (
        <div className="sm:hidden border-t border-white/10 bg-black/95 px-6 py-4 flex flex-col gap-4 text-sm text-gray-400">
          <button onClick={() => scrollTo("hero")} className="hover:text-white transition text-left">
            Home
          </button>
          <button onClick={() => scrollTo("predict")} className="hover:text-white transition text-left">
            Predict
          </button>
        </div>
      )}
    </nav>
  );
}
