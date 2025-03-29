import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchSimulationData = async () => {
  const response = await api.get('/simulation');
  return response.data;
};

export const updateSimulationParams = async (params: any) => {
  const response = await api.post('/simulation/update', params);
  return response.data;
};