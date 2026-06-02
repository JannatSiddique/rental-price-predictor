"use client";

import { useState, useEffect } from "react";
import { fetchLocations, predictRent } from "@/lib/api";
import type { PredictResponse } from "@/types";

export default function PredictionForm() {
  const [bedrooms, setBedrooms] = useState(2);
  const [washrooms, setWashrooms] = useState(2);
  const [marla, setMarla] = useState(5);
  const [location, setLocation] = useState("");
  const [locations, setLocations] = useState<string[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchLocations()
      .then((data) => {
        setLocations(data);
        if (data.length > 0) setLocation(data[0]);
      })
      .catch(() => setError("Failed to load locations"));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await predictRent({
        bedrooms,
        washrooms,
        marla,
        location,
      });
      setResult(res);
    } catch {
      setError("Failed to get prediction. Make sure the API server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label
              htmlFor="bedrooms"
              className="block text-sm font-medium text-gray-300 mb-1"
            >
              Bedrooms
            </label>
            <input
              id="bedrooms"
              type="number"
              min={0}
              max={10}
              value={bedrooms}
              onChange={(e) => setBedrooms(Number(e.target.value))}
              className="block w-full rounded-lg border border-white/10 bg-white/5 px-4 py-2.5 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none transition"
            />
          </div>

          <div>
            <label
              htmlFor="washrooms"
              className="block text-sm font-medium text-gray-300 mb-1"
            >
              Washrooms
            </label>
              <input
              id="washrooms"
              type="number"
              min={0}
              max={10}
              value={washrooms}
              onChange={(e) => setWashrooms(Number(e.target.value))}
              className="block w-full rounded-lg border border-white/10 bg-white/5 px-4 py-2.5 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none transition"
            />
          </div>

          <div>
            <label
              htmlFor="marla"
              className="block text-sm font-medium text-gray-300 mb-1"
            >
              Area (Marla)
            </label>
              <input
              id="marla"
              type="number"
              step="0.1"
              min={1}
              max={50}
              value={marla}
              onChange={(e) => setMarla(Number(e.target.value))}
              className="block w-full rounded-lg border border-white/10 bg-white/5 px-4 py-2.5 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none transition"
            />
          </div>
        </div>

        <div>
          <label
            htmlFor="location"
              className="block text-sm font-medium text-gray-300 mb-1"
            >
              Location
            </label>
          <select
            id="location"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="block w-full rounded-lg border border-white/10 bg-gray-900 px-4 py-2.5 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none transition"
          >
            {locations.map((loc) => (
              <option key={loc} value={loc} className="bg-gray-900 text-white">
                {loc}
              </option>
            ))}
          </select>
        </div>

        {error && (
          <div className="rounded-lg bg-red-950/50 border border-red-500/30 text-red-400 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-gradient-to-r from-purple-600 to-purple-500 px-4 py-3 text-white font-semibold hover:from-purple-500 hover:to-purple-400 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg shadow-purple-500/25"
        >
          {loading ? "Predicting..." : "Predict Rent"}
        </button>
      </form>

      {result && (
        <div className="mt-8 rounded-xl border border-purple-500/20 bg-purple-950/30 p-6 text-center">
          <p className="text-sm text-purple-400 font-medium">Estimated Rent</p>
          <p className="text-3xl font-bold text-purple-300 mt-1">
            PKR {result.predicted_price_pkr.toLocaleString()}
          </p>
          <p className="text-lg text-purple-400 font-medium mt-1">
            ({result.formatted})
          </p>
          <p className="text-sm text-purple-500 mt-3">
            {result.bedrooms} bed &middot; {result.washrooms} bath &middot;{" "}
            {result.marla} Marla &middot; {result.location}
          </p>
        </div>
      )}
    </div>
  );
}
