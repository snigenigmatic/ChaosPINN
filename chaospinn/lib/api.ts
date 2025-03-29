import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchSimulationData = async () => {
  try {
    const response = await api.get('/simulation');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch simulation data:', error);
    return null;
  }
};

export const updateSimulationParams = async (params: any) => {
  try {
    const response = await api.post('/simulation/update', params);
    return response.data;
  } catch (error) {
    console.error('Failed to update simulation parameters:', error);
    return null;
  }
};