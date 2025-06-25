import PDFUploader from '@/components/PDFUploader'

export default function Upload() {
  const handleUploadSuccess = (response: any) => {
    console.log('Upload successful:', response)
    // You can add additional logic here, like redirecting to a Q&A page
  }

  const handleUploadError = (error: string) => {
    console.error('Upload failed:', error)
    // You can add additional error handling here
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Upload PDF Document
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Upload your PDF documents to extract content and enable AI-powered question answering.
            The system will process your document and create embeddings for intelligent search.
          </p>
        </div>
        
        <PDFUploader 
          onUploadSuccess={handleUploadSuccess}
          onUploadError={handleUploadError}
        />
      </div>
    </div>
  )
} 