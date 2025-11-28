/**
 * HS2 Dashboard - Tabbed Workspace
 * Professional 4-tab layout for Overview, GIS, BIM, and Progress Verification
 */

import React from 'react';
import { HS2TabbedLayout } from '../components/hs2/layout/HS2TabbedLayout';
import { HS2OverviewTab } from '../components/hs2/overview/HS2OverviewTab';
import { HS2GISTab } from '../components/hs2/gis/HS2GISTab';
import { HS2BIMTab } from '../components/hs2/bim/HS2BIMTab';
import { ProgressVerificationTab } from '../components/hs2/progress/ProgressVerificationTab';

const HS2DashboardNew: React.FC = () => {
  return (
    <HS2TabbedLayout>
      {[
        <HS2OverviewTab key="overview" />,
        <HS2GISTab key="gis" />,
        <HS2BIMTab key="bim" />,
        <ProgressVerificationTab key="progress" />
      ]}
    </HS2TabbedLayout>
  );
};

export default HS2DashboardNew;
