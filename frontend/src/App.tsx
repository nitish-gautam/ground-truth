import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, CssBaseline } from '@mui/material';
import HS2DashboardNew from './pages/HS2DashboardNew';
import HS2AssetList from './pages/HS2AssetList';
import HS2AssetDetail from './pages/HS2AssetDetail';
import { HS2FileUpload } from './pages/HS2FileUpload';
import { HS2NoiseMonitoring } from './pages/HS2NoiseMonitoring';
import hs2Theme from './theme';
import './App.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000 // 5 minutes
    }
  }
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={hs2Theme}>
        <CssBaseline />
        <BrowserRouter>
          <Routes>
            {/* Default route redirects to HS2 Dashboard */}
            <Route path="/" element={<Navigate to="/hs2" replace />} />
            
            {/* HS2 Routes */}
            <Route path="/hs2" element={<HS2DashboardNew />} />
            <Route path="/hs2/assets" element={<HS2AssetList />} />
            <Route path="/hs2/assets/:assetId" element={<HS2AssetDetail />} />
            <Route path="/hs2/upload" element={<HS2FileUpload />} />
            <Route path="/hs2/monitoring" element={<HS2NoiseMonitoring />} />

            {/* Catch-all route */}
            <Route path="*" element={<Navigate to="/hs2" replace />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
