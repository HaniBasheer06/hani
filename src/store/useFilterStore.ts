import { create } from 'zustand';
import type { PredictionResponse } from '../types/api';

interface FilterState {
  state: string;
  district: string;
  search: string;
  selectedDepositId: string | null;
  mapCoordinate: { lat: number; lng: number } | null;
  predictionOpen: boolean;
  prediction: PredictionResponse | null;
  setFilter: (key: 'state' | 'district' | 'search', value: string) => void;
  selectDeposit: (id: string | null) => void;
  setMapCoordinate: (coordinate: { lat: number; lng: number } | null) => void;
  setPredictionOpen: (open: boolean) => void;
  setPrediction: (prediction: PredictionResponse | null) => void;
  reset: () => void;
}

const initial = { state: '', district: '', search: '', selectedDepositId: null, mapCoordinate: null, predictionOpen: false, prediction: null };

export const useFilterStore = create<FilterState>((set) => ({
  ...initial,
  setFilter: (key, value) => set((current) => {
    if (key === 'state') return { ...current, state: value, district: '' };
    if (key === 'district') return { ...current, district: value };
    return { ...current, search: value };
  }),
  selectDeposit: (selectedDepositId) => set({ selectedDepositId }),
  setMapCoordinate: (mapCoordinate) => set({ mapCoordinate }),
  setPredictionOpen: (predictionOpen) => set({ predictionOpen }),
  setPrediction: (prediction) => set({ prediction }),
  reset: () => set(initial),
}));
