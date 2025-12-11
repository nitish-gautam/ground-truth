/**
 * HS2 Dashboard - Tabbed Workspace
 * Professional 7-tab layout for Overview, GIS, BIM, LiDAR, Hyperspectral, Integrated Demo, and Progress Verification
 */

import React from 'react';
import { HS2TabbedLayout } from '../components/hs2/layout/HS2TabbedLayout';
import { HS2OverviewTab } from '../components/hs2/overview/HS2OverviewTab';
import { HS2GISTab } from '../components/hs2/gis/HS2GISTab';
import { HS2BIMTab } from '../components/hs2/bim/HS2BIMTab';
import { LidarViewerTab } from '../components/hs2/lidar/LidarViewerTab';
import { HyperspectralViewerTab } from '../components/hs2/hyperspectral/HyperspectralViewerTab';
import { RealDataDashboard } from '../components/hs2/demo/RealDataDashboard';
import { ProgressVerificationTab } from '../components/hs2/progress/ProgressVerificationTab';

const HS2DashboardNew: React.FC = () => {
  return (
    <HS2TabbedLayout>
      {[
        <HS2OverviewTab key="overview" />,
        <HS2GISTab key="gis" />,
        <HS2BIMTab key="bim" />,
        <LidarViewerTab key="lidar" />,
        <HyperspectralViewerTab key="hyperspectral" />,
        <RealDataDashboard key="demo" />,
        <ProgressVerificationTab key="progress" />
      ]}
    </HS2TabbedLayout>
  );
};

export default HS2DashboardNew;
