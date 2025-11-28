/**
 * FileUpload Component
 * Drag & drop file upload with progress tracking
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import axios from 'axios';

interface FileWithProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  fileId?: string;
  error?: string;
}

interface FileUploadProps {
  onFileUploaded?: (fileId: string) => void;
  acceptedFormats?: string[];
  maxFileSize?: number; // in MB
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileUploaded,
  acceptedFormats = [
    '.xlsx', '.xls',  // Excel
    '.pdf',           // PDF
    '.ifc',           // BIM
    '.dwg', '.dxf',   // CAD
    '.shp',           // GIS
    '.laz', '.las',   // LiDAR
    '.sgy', '.segy'   // GPR
  ],
  maxFileSize = 500 // 500MB default
}) => {
  const [files, setFiles] = useState<FileWithProgress[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle drop event
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  }, []);

  // Handle file input change
  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  }, []);

  // Validate and add files to upload queue
  const handleFiles = useCallback((newFiles: File[]) => {
    setError(null);
    const maxSizeBytes = maxFileSize * 1024 * 1024;

    const validatedFiles: FileWithProgress[] = newFiles.map(file => {
      // Check file size
      if (file.size > maxSizeBytes) {
        return {
          file,
          progress: 0,
          status: 'error' as const,
          error: `File size exceeds ${maxFileSize}MB limit`
        };
      }

      // Check file extension
      const extension = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!acceptedFormats.some(format => format.toLowerCase() === extension)) {
        return {
          file,
          progress: 0,
          status: 'error' as const,
          error: 'File format not supported'
        };
      }

      return {
        file,
        progress: 0,
        status: 'pending' as const
      };
    });

    setFiles(prev => [...prev, ...validatedFiles]);

    // Auto-upload valid files
    validatedFiles.forEach((fileWithProgress, index) => {
      if (fileWithProgress.status === 'pending') {
        uploadFile(fileWithProgress, files.length + index);
      }
    });
  }, [acceptedFormats, maxFileSize, files.length]);

  // Upload file to backend
  const uploadFile = async (fileWithProgress: FileWithProgress, index: number) => {
    const formData = new FormData();
    formData.append('file', fileWithProgress.file);

    try {
      // Update status to uploading
      setFiles(prev => {
        const updated = [...prev];
        updated[index] = { ...updated[index], status: 'uploading' };
        return updated;
      });

      const response = await axios.post('/api/v1/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;

          setFiles(prev => {
            const updated = [...prev];
            updated[index] = { ...updated[index], progress: percentCompleted };
            return updated;
          });
        }
      });

      // Upload successful
      const fileId = response.data.file_id;
      setFiles(prev => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          status: 'success',
          progress: 100,
          fileId
        };
        return updated;
      });

      // Notify parent component
      if (onFileUploaded && fileId) {
        onFileUploaded(fileId);
      }

    } catch (err: any) {
      // Upload failed
      const errorMessage = err.response?.data?.detail || 'Upload failed';
      setFiles(prev => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          status: 'error',
          error: errorMessage
        };
        return updated;
      });
    }
  };

  // Remove file from list
  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Clear all files
  const clearAll = () => {
    setFiles([]);
    setError(null);
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'uploading':
        return <CloudUploadIcon color="primary" />;
      default:
        return <InsertDriveFileIcon />;
    }
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Drag & Drop Zone */}
      <Paper
        sx={{
          p: 4,
          border: '2px dashed',
          borderColor: dragActive ? 'primary.main' : 'grey.300',
          bgcolor: dragActive ? 'action.hover' : 'background.paper',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'action.hover'
          }
        }}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <CloudUploadIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Drag & Drop Files Here
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          or click to browse files
        </Typography>

        <input
          type="file"
          multiple
          accept={acceptedFormats.join(',')}
          onChange={handleFileInput}
          style={{ display: 'none' }}
          id="file-upload-input"
        />
        <label htmlFor="file-upload-input">
          <Button variant="contained" component="span">
            Choose Files
          </Button>
        </label>

        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Supported formats: {acceptedFormats.join(', ')}
          </Typography>
          <br />
          <Typography variant="caption" color="text.secondary">
            Max file size: {maxFileSize}MB
          </Typography>
        </Box>
      </Paper>

      {/* File List */}
      {files.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Uploaded Files ({files.length})
            </Typography>
            <Button size="small" onClick={clearAll}>
              Clear All
            </Button>
          </Box>

          <List>
            {files.map((fileWithProgress, index) => (
              <ListItem
                key={index}
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  mb: 1
                }}
                secondaryAction={
                  <IconButton edge="end" onClick={() => removeFile(index)}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  {getStatusIcon(fileWithProgress.status)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2">
                        {fileWithProgress.file.name}
                      </Typography>
                      <Chip
                        label={fileWithProgress.status}
                        size="small"
                        color={
                          fileWithProgress.status === 'success' ? 'success' :
                          fileWithProgress.status === 'error' ? 'error' :
                          'default'
                        }
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        {(fileWithProgress.file.size / 1024 / 1024).toFixed(2)} MB
                      </Typography>
                      {fileWithProgress.error && (
                        <Typography variant="caption" color="error" display="block">
                          {fileWithProgress.error}
                        </Typography>
                      )}
                      {fileWithProgress.status === 'uploading' && (
                        <LinearProgress
                          variant="determinate"
                          value={fileWithProgress.progress}
                          sx={{ mt: 1 }}
                        />
                      )}
                      {fileWithProgress.fileId && (
                        <Typography variant="caption" color="text.secondary" display="block">
                          File ID: {fileWithProgress.fileId}
                        </Typography>
                      )}
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );
};
