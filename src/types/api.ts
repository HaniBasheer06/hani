export type Nullable<T> = T | null;

export interface HealthResponse {
  status: string;
  dataset_rows: number;
  inventory_rows?: number;
  dataset_file: string;
}

export interface SummaryResponse {
  total_records: number;
  states: number;
  districts: number;
  total_annual_production_tonnes: number;
  average_grade_pct: Nullable<number>;
  year_range: string[];
  inventory_records?: number;
  raw_records?: number;
  training_rows?: number;
  synthetic_training_rows?: number;
  training_r2?: number;
  cross_validation_r2_mean?: number;
  cross_validation_r2_std?: number;
  grouped_cross_validation_r2_mean?: number | null;
  validation_note?: string;
  original_training_rows?: number;
  real_observations?: number;
  synthetic_observations?: number;
  year_basis?: string;
  soil_types?: string[];
  provenance_note?: string;
}

export interface DepositRecord {
  State: Nullable<string>;
  District: Nullable<string>;
  Deposit_ID: Nullable<string>;
  Latitude: Nullable<number>;
  Longitude: Nullable<number>;
  Reserves_tonnes?: Nullable<number>;
  Annual_Production_tonnes: Nullable<number>;
  Grade_pct?: Nullable<number>;
  Soil_Type: Nullable<string>;
  Road_Accessibility: Nullable<string>;
  [key: string]: unknown;
}

export interface DepositsResponse {
  items: DepositRecord[];
  page: number;
  limit: number;
  total: number;
}

export interface TrendPoint {
  year: string | number;
  production_tonnes: number;
}

export interface TrendResponse {
  series: TrendPoint[];
}

export interface MapPoint {
  deposit_id: Nullable<string>;
  state: Nullable<string>;
  district: Nullable<string>;
  latitude: number;
  longitude: number;
  production_tonnes: Nullable<number>;
  grade_pct: Nullable<number>;
}

export interface MapDepositsResponse {
  points: MapPoint[];
}

export interface ModelMetricsResponse {
  model: string;
  target: string;
  mae: number;
  rmse: number;
  r2: number;
  training_rows?: number;
  cleaned_rows?: number;
  original_training_rows?: number;
  synthetic_training_rows?: number;
  test_rows?: number;
  test_size?: number;
  charts: {
    actual_vs_predicted: string;
    feature_importance: string;
  };
}

export interface PredictionRequest {
  Latitude: number;
  Longitude: number;
  State?: string;
  District?: string;
  Annual_Precip_mm?: number;
  Soil_Type?: string;
  Topo_Slope_deg?: number;
  Road_Accessibility?: string;
  Distance_to_Port_km?: number;
  Avg_Temperature_C?: number;
}

export interface ContributingFactor {
  feature: string;
  value: string;
  impact: 'increased' | 'decreased';
  contribution_tonnes: number;
  reason: string;
}

export interface PredictionResponse {
  predicted_annual_production_tonnes: number;
  feasibility_rating: string;
  accuracy_metrics: {
    model_r2_score: number;
    r2_percentage: string;
  };
  hydrated_from_district?: Nullable<string>;
  nearest_port?: Nullable<string>;
  explanation: {
    top_contributing_factors: ContributingFactor[];
  };
  training_data_coverage?: {
    level: 'High' | 'Moderate' | 'Low';
    score: number;
    warning: string | null;
    out_of_range_features: Array<{ feature: string; value: number; training_range: [number, number] }>;
    unknown_categories: string[];
  };
  input_features_used?: Record<string, unknown>;
}
