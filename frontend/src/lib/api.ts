import type { PredictRequest, PredictResponse, LocationsResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchLocations(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/locations`);
  const data: LocationsResponse = await res.json();
  return data.locations;
}

export async function predictRent(
  params: PredictRequest
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error("Prediction failed");
  }
  return res.json();
}
