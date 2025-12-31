/**
 * HS2 Dashboard - Tabbed Workspace
 * Streamlined 4-tab layout: Dashboard, GIS Map, Data Viewers, Analytics & Reports
 * Reduced cognitive load from 7 tabs to 4 with smart grouping
 */

import React from 'react';
import { HS2TabbedLayout } from '../components/hs2/layout/HS2TabbedLayout';
import { HS2OverviewTab } from '../components/hs2/overview/HS2OverviewTab';
import { HS2GISTab } from '../components/hs2/gis/HS2GISTab';
import { DataViewersTab } from '../components/hs2/viewers/DataViewersTab';
import { AnalyticsTab } from '../components/hs2/analytics/AnalyticsTab';

const HS2DashboardNew: React.FC = () => {
  return (
    <HS2TabbedLayout>
      {[
        <HS2OverviewTab key="overview" />,
        <HS2GISTab key="gis" />,
        <DataViewersTab key="viewers" />,
        <AnalyticsTab key="analytics" />
      ]}
    </HS2TabbedLayout>
  );
};

export default HS2DashboardNew;
