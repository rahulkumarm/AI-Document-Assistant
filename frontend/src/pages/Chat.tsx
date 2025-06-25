import { useState } from 'react'
import ChatBox from '@/components/ChatBox'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { MessageCircle, FileText } from 'lucide-react'

export default function Chat() {
  const [documentId, setDocumentId] = useState('')
  const [showDocumentInput, setShowDocumentInput] = useState(false)

  const handleChatError = (error: string) => {
    console.error('Chat error:', error)
    // You can add additional error handling here, like showing a toast
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            AI Chat Assistant
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto mb-6">
            Chat with our AI assistant. You can ask general questions or specific questions about your uploaded documents.
          </p>
          
          {/* Document ID Input */}
          <div className="max-w-md mx-auto mb-6">
            {!showDocumentInput ? (
              <Button 
                variant="outline" 
                onClick={() => setShowDocumentInput(true)}
                className="mb-4"
              >
                <FileText className="w-4 h-4 mr-2" />
                Chat about a specific document
              </Button>
            ) : (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Document-Specific Chat</CardTitle>
                  <CardDescription>
                    Enter the document ID from your uploaded PDF to ask questions about it
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Input
                    placeholder="Enter document ID (e.g., doc_abc123)"
                    value={documentId}
                    onChange={(e) => setDocumentId(e.target.value)}
                  />
                  <div className="flex space-x-2">
                    <Button 
                      onClick={() => setShowDocumentInput(false)}
                      variant="outline"
                      size="sm"
                    >
                      Cancel
                    </Button>
                    <Button 
                      onClick={() => setShowDocumentInput(false)}
                      size="sm"
                      disabled={!documentId.trim()}
                    >
                      Start Document Chat
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
        
        <ChatBox 
          documentId={documentId || undefined}
          onError={handleChatError}
        />
        
        {/* Info Cards */}
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageCircle className="w-5 h-5" />
                General Chat
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Ask me anything! I can help with general questions, explanations, coding, writing, and more.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Document Q&A
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Upload a PDF first, then use the document ID to ask specific questions about your document's content.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 