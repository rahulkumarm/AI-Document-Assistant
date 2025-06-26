import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

interface PDFUploaderProps {
  onUploadSuccess?: (response: any) => void
  onUploadError?: (error: string) => void
  onFileSelected?: (file: File | null) => void
}

interface UploadStatus {
  status: 'idle' | 'uploading' | 'success' | 'error'
  message?: string
  response?: any
}

export default function PDFUploader({ onUploadSuccess, onUploadError, onFileSelected }: PDFUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({ status: 'idle' })
  const [saveToVectorStore, setSaveToVectorStore] = useState(true)
  const { toast } = useToast()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      if (file.type === 'application/pdf') {
        setSelectedFile(file)
        setUploadStatus({ status: 'idle' })
        if (onFileSelected) {
          onFileSelected(file)
        }
      } else {
        setUploadStatus({ 
          status: 'error', 
          message: 'Please select a PDF file only.' 
        })
        toast({
          title: "Invalid File Type",
          description: "Please select a PDF file only.",
          variant: "destructive",
        })
        if (onFileSelected) {
          onFileSelected(null)
        }
      }
    }
  }, [toast, onFileSelected])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    multiple: false
  })

  const handleSubmit = async () => {
    if (!selectedFile) {
      setUploadStatus({ 
        status: 'error', 
        message: 'Please select a PDF file first.' 
      })
      toast({
        title: "No File Selected",
        description: "Please select a PDF file first.",
        variant: "destructive",
      })
      return
    }

    setUploadStatus({ status: 'uploading', message: 'Uploading PDF...' })
    
    // Show uploading toast
    toast({
      title: "Uploading...",
      description: `Processing ${selectedFile.name}...`,
    })

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('save_to_vector_store', saveToVectorStore.toString())

      // Create AbortController for timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout

      // Temporary hardcoded URL for testing - replace with environment variable
      const apiUrl = import.meta.env.VITE_API_URL || 'https://ai-document-assistant-production.up.railway.app'
      console.log('Using API URL:', apiUrl) // Debug log
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      setUploadStatus({ 
        status: 'success', 
        message: 'PDF uploaded and processed successfully!',
        response: data
      })

      // Show success toast
      toast({
        title: "Upload Successful!",
        description: `${selectedFile.name} has been processed and is ready for Q&A.`,
      })

      if (onUploadSuccess) {
        onUploadSuccess(data)
      }
    } catch (error: any) {
      console.error('Upload error:', error)
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack,
        cause: error.cause
      })
      
      let errorMessage = 'Failed to upload PDF. Please try again.'
      
      if (error.name === 'AbortError') {
        errorMessage = 'The request timed out. Please try a smaller file or check your connection.'
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Network error: Unable to connect to the server. Please check if the backend is running on port 8000.'
      } else if (error.message?.includes('413')) {
        errorMessage = 'File is too large. Please select a file smaller than 10MB.'
      } else if (error.message?.includes('422')) {
        errorMessage = 'Invalid file format. Please ensure you\'re uploading a valid PDF.'
      } else if (error.message) {
        errorMessage = error.message
      }

      setUploadStatus({ 
        status: 'error', 
        message: errorMessage
      })

      // Show error toast
      toast({
        title: "Upload Failed",
        description: errorMessage,
        variant: "destructive",
      })

      if (onUploadError) {
        onUploadError(errorMessage)
      }
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setUploadStatus({ status: 'idle' })
    if (onFileSelected) {
      onFileSelected(null)
    }
  }

  const getStatusIcon = () => {
    switch (uploadStatus.status) {
      case 'uploading':
        return <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-600" />
      default:
        return null
    }
  }

  const getStatusColor = () => {
    switch (uploadStatus.status) {
      case 'uploading':
        return 'text-blue-600'
      case 'success':
        return 'text-green-600'
      case 'error':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Upload PDF Document
        </CardTitle>
        <CardDescription>
          Upload a PDF file to extract text and generate embeddings for AI-powered Q&A
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drag and Drop Area */}
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-blue-500 bg-blue-50' 
              : selectedFile 
                ? 'border-green-500 bg-green-50' 
                : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            }
          `}
        >
          <input {...getInputProps()} />
          
          {selectedFile ? (
            <div className="space-y-2">
              <FileText className="w-12 h-12 mx-auto text-green-600" />
              <div>
                <p className="font-medium text-green-700">{selectedFile.name}</p>
                <p className="text-sm text-green-600">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  handleReset()
                }}
                className="mt-2"
              >
                <X className="w-4 h-4 mr-1" />
                Remove
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="w-12 h-12 mx-auto text-gray-400" />
              <div>
                <p className="text-lg font-medium text-gray-700">
                  {isDragActive ? 'Drop the PDF file here' : 'Drag & drop a PDF file here'}
                </p>
                <p className="text-sm text-gray-500">
                  or click to select a file (max 10MB)
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Options */}
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="saveToVectorStore"
            checked={saveToVectorStore}
            onChange={(e) => setSaveToVectorStore(e.target.checked)}
            className="rounded border-gray-300"
          />
          <label htmlFor="saveToVectorStore" className="text-sm text-gray-700">
            Save to vector store for Q&A (recommended)
          </label>
        </div>

        {/* Status Message */}
        {uploadStatus.message && (
          <div className={`flex items-center gap-2 p-3 rounded-lg ${
            uploadStatus.status === 'success' ? 'bg-green-50' :
            uploadStatus.status === 'error' ? 'bg-red-50' :
            'bg-blue-50'
          }`}>
            {getStatusIcon()}
            <span className={`text-sm ${getStatusColor()}`}>
              {uploadStatus.message}
            </span>
          </div>
        )}

        {/* Success Details */}
        {uploadStatus.status === 'success' && uploadStatus.response && (
          <div className="bg-green-50 p-4 rounded-lg space-y-2">
            <h4 className="font-medium text-green-800">Upload Details:</h4>
            <div className="text-sm text-green-700 space-y-1">
              <p><strong>Document ID:</strong> {uploadStatus.response.document_id}</p>
              <p><strong>Pages:</strong> {uploadStatus.response.total_pages}</p>
              <p><strong>Total Chunks:</strong> {uploadStatus.response.total_chunks}</p>
              {uploadStatus.response.embeddings_generated && (
                <p><strong>Embeddings:</strong> Generated and saved to vector store</p>
              )}
            </div>
          </div>
        )}

        {/* Submit Button */}
        <Button 
          onClick={handleSubmit}
          disabled={!selectedFile || uploadStatus.status === 'uploading'}
          className="w-full"
          size="lg"
        >
          {uploadStatus.status === 'uploading' ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="w-4 h-4 mr-2" />
              Upload and Process PDF
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  )
}