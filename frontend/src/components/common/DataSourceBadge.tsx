/**
 * DataSourceBadge Component
 * Clearly indicates whether data is live/demo/synthetic
 */

import React from 'react';
import { Chip, ChipProps, Tooltip } from '@mui/material';
import { CheckCircle, Science, Warning } from '@mui/icons-material';

export type DataSource = 'live' | 'demo' | 'synthetic';

interface DataSourceBadgeProps extends Omit<ChipProps, 'label' | 'color'> {
  source: DataSource;
  showTooltip?: boolean;
}

const SOURCE_CONFIG: Record<DataSource, {
  label: string;
  color: ChipProps['color'];
  icon: React.ReactElement;
  tooltip: string;
}> = {
  live: {
    label: 'Live Data',
    color: 'success',
    icon: <CheckCircle fontSize="small" />,
    tooltip: 'Real-time data from production systems'
  },
  demo: {
    label: 'Demo Data',
    color: 'info',
    icon: <Science fontSize="small" />,
    tooltip: 'Representative sample data for demonstration purposes'
  },
  synthetic: {
    label: 'Synthetic Data',
    color: 'warning',
    icon: <Warning fontSize="small" />,
    tooltip: 'Computer-generated data for testing and development'
  }
};

export const DataSourceBadge: React.FC<DataSourceBadgeProps> = ({
  source,
  showTooltip = true,
  size = 'small',
  ...chipProps
}) => {
  const config = SOURCE_CONFIG[source];

  const chip = (
    <Chip
      label={config.label}
      color={config.color}
      size={size}
      icon={config.icon}
      {...chipProps}
    />
  );

  if (showTooltip) {
    return (
      <Tooltip title={config.tooltip} arrow>
        {chip}
      </Tooltip>
    );
  }

  return chip;
};

export default DataSourceBadge;
