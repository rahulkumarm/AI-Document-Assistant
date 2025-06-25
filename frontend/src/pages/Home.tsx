import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, MessageCircleQuestion, CheckCircle, FileText } from "lucide-react"
import { useState } from "react"
import PDFUploader from "@/components/PDFUploader"
import ChatBox from "@/components/ChatBox"
import PDFViewer from "@/components/PDFViewer"

export default function Home() {
  const [fileId, setFileId] = useState<string | undefined>()
  const [uploadedFileName, setUploadedFileName] = useState<string>("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleUploadSuccess = (response: any) => {
    console.log('Upload successful:', response)
    setFileId(response.document_id)
    setUploadedFileName(response.filename || "Document")
  }

  const handleUploadError = (error: string) => {
    console.error('Upload failed:', error)
  }

  const handleChatError = (error: string) => {
    console.error('Chat error:', error)
  }

  const handleFileSelected = (file: File | null) => {
    setSelectedFile(file)
  }

  const handleReset = () => {
    setFileId(undefined)
    setUploadedFileName("")
    setSelectedFile(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            AI Document Assistant
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Upload your PDF documents and ask questions to get intelligent answers 
            based on the content using advanced RAG technology.
          </p>
        </div>

        {/* Main Content - Three Column Layout */}
        <div className="max-w-full mx-auto mb-20">
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Left Column - PDF Uploader */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Document
                </h2>
                {fileId && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleReset}
                  >
                    Upload New Document
                  </Button>
                )}
              </div>
              
              {fileId ? (
                <Card className="border-green-200 bg-green-50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-800">
                      <FileText className="w-5 h-5" />
                      Document Ready
                    </CardTitle>
                    <CardDescription className="text-green-700">
                      Your document has been processed and is available for Q&A
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm text-green-700">
                      <p><strong>File:</strong> {uploadedFileName}</p>
                      <p><strong>Document ID:</strong> {fileId}</p>
                      <p>You can now ask questions about this document in the chat.</p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <PDFUploader 
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={handleUploadError}
                  onFileSelected={handleFileSelected}
                />
              )}
            </div>

            {/* Middle Column - PDF Viewer */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  PDF Preview
                </h2>
                {selectedFile && (
                  <div className="text-sm text-gray-600">
                    {selectedFile.name}
                  </div>
                )}
              </div>
              
              <div className="h-[600px] border rounded-lg overflow-hidden bg-white">
                <PDFViewer 
                  file={selectedFile}
                  className="h-full w-full"
                />
              </div>
            </div>

            {/* Right Column - Chat Box */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                  <MessageCircleQuestion className="w-5 h-5" />
                  Ask Questions
                </h2>
                {fileId && (
                  <div className="flex items-center gap-2 text-sm text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    Document loaded
                  </div>
                )}
              </div>
              
              {fileId ? (
                <ChatBox 
                  fileId={fileId}
                  onError={handleChatError}
                />
              ) : (
                <Card className="h-[600px] flex items-center justify-center">
                  <CardContent className="text-center">
                    <MessageCircleQuestion className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p className="text-gray-500 mb-2">No document uploaded yet</p>
                    <p className="text-sm text-gray-400">
                      Upload a PDF document first to start asking questions
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-20">
          <h2 className="text-2xl font-semibold text-center mb-12 text-gray-900">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="text-center">
              <div className="mx-auto mb-6 p-4 bg-blue-100 rounded-full w-fit">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="font-semibold mb-3 text-gray-900 text-lg">1. Upload PDF</h3>
              <p className="text-gray-600">
                Upload your PDF document. Our system will extract and process the text content, 
                creating searchable chunks for optimal AI understanding.
              </p>
            </div>
            <div className="text-center">
              <div className="mx-auto mb-6 p-4 bg-green-100 rounded-full w-fit">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
              </div>
              <h3 className="font-semibold mb-3 text-gray-900 text-lg">2. AI Processing</h3>
              <p className="text-gray-600">
                Advanced AI models generate embeddings and store your document content 
                in a vector database for intelligent retrieval and context understanding.
              </p>
            </div>
            <div className="text-center">
              <div className="mx-auto mb-6 p-4 bg-purple-100 rounded-full w-fit">
                <MessageCircleQuestion className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="font-semibold mb-3 text-gray-900 text-lg">3. Ask Questions</h3>
              <p className="text-gray-600">
                Ask natural language questions about your document. Get accurate, 
                contextual answers powered by retrieval-augmented generation (RAG).
              </p>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-16 text-center">
          <div className="bg-white rounded-lg shadow-sm border p-8 max-w-3xl mx-auto">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Ready to get started?
            </h3>
            <p className="text-gray-600 mb-6">
              Upload a PDF document on the left and start asking questions on the right. 
              The AI will provide intelligent answers based on your document's content.
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-500">
              <span className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                PDF files up to 10MB
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Instant text extraction
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                AI-powered Q&A
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Contextual answers
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 