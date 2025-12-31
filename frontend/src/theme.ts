/**
 * HS2 Brand Theme
 * Based on brand guidelines with Galano Grotesque typography and custom color palette
 */
import { createTheme } from '@mui/material/styles';

const hs2Theme = createTheme({
  palette: {
    primary: {
      main: '#012A39', // Brand primary color
      light: '#023d52',
      dark: '#011d28',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#019C4B', // Brand secondary color
      light: '#02b857',
      dark: '#017339',
      contrastText: '#FFFFFF',
    },
    error: {
      main: '#D32F2F', // Material Design red-600 (softer than pure red)
      light: '#EF5350',
      dark: '#C62828',
    },
    warning: {
      main: '#FF8500',
      light: '#ff9d33',
      dark: '#cc6a00',
    },
    info: {
      main: '#0288D1', // Material Design light-blue-600 (WCAG AA compliant)
      light: '#4FC3F7',
      dark: '#01579B',
    },
    success: {
      main: '#00A896', // Distinct teal (different from secondary green)
      light: '#26C6B8',
      dark: '#007A6E',
    },
    background: {
      default: '#FFFFFF',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#012A39',
      secondary: '#4A5F6B',
    },
  },
  typography: {
    fontFamily: [
      'Galano Grotesque',
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
      color: '#012A39',
    },
    h2: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 700,
      fontSize: '2rem',
      lineHeight: 1.3,
      color: '#012A39',
    },
    h3: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.75rem',
      lineHeight: 1.3,
      color: '#012A39',
    },
    h4: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
      color: '#012A39',
    },
    h5: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
      color: '#012A39',
    },
    h6: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.125rem',
      lineHeight: 1.4,
      color: '#012A39',
    },
    body1: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 400,
      fontSize: '1rem',
      lineHeight: 1.5,
    },
    body2: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 400,
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    button: {
      fontFamily: 'Galano Grotesque, Inter, sans-serif',
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 24px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(1, 42, 57, 0.15)',
          },
        },
        containedSecondary: {
          backgroundColor: '#017339', // secondary.dark
          color: '#FFFFFF',
          '&:hover': {
            backgroundColor: '#019C4B', // secondary.main
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(1, 42, 57, 0.08)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(1, 42, 57, 0.08)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
        sizeSmall: {
          height: 24,
          fontSize: '0.75rem',
          '& .MuiChip-label': {
            paddingLeft: 8,
            paddingRight: 8,
          },
        },
        outlined: {
          borderColor: '#019C4B', // secondary.main
          color: '#012A39', // primary.main
          '&:hover': {
            backgroundColor: 'rgba(1, 156, 75, 0.08)', // secondary.light with opacity
          },
        },
        filled: {
          backgroundColor: '#019C4B', // secondary.main
          color: '#FFFFFF',
          '&:hover': {
            backgroundColor: '#017339', // secondary.dark
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: 'rgba(0, 0, 0, 0.54)', // grey color
          '&:hover': {
            backgroundColor: '#023d52', // primary.light
            color: '#FFFFFF', // primary.contrastText
          },
          '&.selected': {
            backgroundColor: '#011d28', // primary.dark
            color: '#FFFFFF', // primary.contrastText
          },
        },
      },
    },
  },
});

export default hs2Theme;
