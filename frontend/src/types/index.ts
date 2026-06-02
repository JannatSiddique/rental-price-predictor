export interface PredictRequest {
  bedrooms: number;
  washrooms: number;
  marla: number;
  location: string;
}

export interface PredictResponse {
  predicted_price_pkr: number;
  formatted: string;
  bedrooms: number;
  washrooms: number;
  marla: number;
  location: string;
}

export interface LocationsResponse {
  locations: string[];
}
