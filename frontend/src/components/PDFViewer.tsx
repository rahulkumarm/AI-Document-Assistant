import { useState, useCallback, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight, FileText, ZoomIn, ZoomOut, RotateCw } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

// Import React PDF styles directly
import 'react-pdf/dist/esm/Page/AnnotationLayer.css'
import 'react-pdf/dist/esm/Page/TextLayer.css'

// Use local worker file to avoid CORS issues
pdfjs.GlobalWorkerOptions.workerSrc = '/js/pdf.worker.min.js'

interface PDFViewerProps {
  file: File | string | null
  className?: string
}

export default function PDFViewer({ file, className = '' }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [scale, setScale] = useState<number>(1.0)
  const [rotation, setRotation] = useState<number>(0)
  const [loading, setLoading] = useState<boolean>(true)
  const [containerWidth, setContainerWidth] = useState<number>(0)
  const { toast } = useToast()

  // Calculate scale based on container width
  const calculateScale = useCallback((pageWidth: number, containerWidth: number) => {
    if (!pageWidth || !containerWidth) return 1.0
    // Account for padding (32px total: 16px on each side)
    const availableWidth = containerWidth - 32
    const calculatedScale = availableWidth / pageWidth
    // Limit scale between 0.1 and 3.0
    return Math.min(Math.max(calculatedScale, 0.1), 3.0)
  }, [])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setPageNumber(1)
    setLoading(false)
    console.log(`PDF loaded with ${numPages} pages`)
    console.log('PDF container should now be scrollable')
  }, [])

  const onDocumentLoadError = useCallback((error: Error) => {
    console.error('Error loading PDF:', error)
    setLoading(false)
    toast({
      title: "PDF Load Error",
      description: "Failed to load the PDF file. Please try again.",
      variant: "destructive",
    })
  }, [toast])

  const onPageLoadError = useCallback((error: Error) => {
    console.error('Error loading page:', error)
    toast({
      title: "Page Load Error", 
      description: "Failed to load the PDF page. Please try again.",
      variant: "destructive",
    })
  }, [toast])

  const onPageRenderSuccess = useCallback((page: any) => {
    console.log('Page rendered successfully:', {
      pageNumber: page.pageNumber,
      width: page.width,
      height: page.height,
      scale: scale,
      actualWidth: page.width * scale,
      actualHeight: page.height * scale
    })
    
    // Get container width and calculate appropriate scale
    const scrollContainer = document.querySelector('.overflow-y-auto')
    if (scrollContainer && page.width) {
      const containerRect = scrollContainer.getBoundingClientRect()
      const newContainerWidth = containerRect.width
      
      if (newContainerWidth !== containerWidth) {
        setContainerWidth(newContainerWidth)
        const newScale = calculateScale(page.width, newContainerWidth)
        if (Math.abs(newScale - scale) > 0.05) { // Only update if significant change
          setScale(newScale)
          console.log('Auto-adjusting scale:', {
            pageWidth: page.width,
            containerWidth: newContainerWidth,
            newScale: newScale
          })
        }
      }
      
      console.log('Scroll container dimensions:', {
        scrollHeight: scrollContainer.scrollHeight,
        clientHeight: scrollContainer.clientHeight,
        scrollTop: scrollContainer.scrollTop,
        canScroll: scrollContainer.scrollHeight > scrollContainer.clientHeight
      })
    }
  }, [scale, containerWidth, calculateScale])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const scrollContainer = document.querySelector('.overflow-y-auto')
      if (scrollContainer) {
        const containerRect = scrollContainer.getBoundingClientRect()
        const newContainerWidth = containerRect.width
        if (newContainerWidth !== containerWidth) {
          setContainerWidth(newContainerWidth)
        }
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [containerWidth])

  const goToPrevPage = () => {
    setPageNumber(prev => Math.max(prev - 1, 1))
  }

  const goToNextPage = () => {
    setPageNumber(prev => Math.min(prev + 1, numPages))
  }

  const zoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3.0))
  }

  const zoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.4))
  }

  const rotate = () => {
    setRotation(prev => (prev + 90) % 360)
  }

  const resetView = () => {
    setScale(1.0)
    setRotation(0)
  }

  if (!file) {
    return (
      <Card className={`w-full ${className}`}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center text-gray-500">
            <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg font-medium">No PDF selected</p>
            <p className="text-sm">Upload a PDF to preview it here</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            PDF Preview
          </CardTitle>
          
          {/* Zoom and Rotation Controls */}
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={zoomOut}
              disabled={scale <= 0.4}
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={zoomIn}
              disabled={scale >= 3.0}
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={rotate}
              title="Rotate"
            >
              <RotateCw className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={resetView}
              title="Reset View"
            >
              Reset
            </Button>
          </div>
        </div>
        
        {/* Page Navigation */}
        {numPages > 0 && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={goToPrevPage}
                disabled={pageNumber <= 1}
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>
              
              <span className="text-sm text-gray-600">
                Page {pageNumber} of {numPages}
              </span>
              
              <Button
                variant="outline"
                size="sm"
                onClick={goToNextPage}
                disabled={pageNumber >= numPages}
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
            
            <div className="text-sm text-gray-500">
              Zoom: {Math.round(scale * 100)}%
            </div>
          </div>
        )}
      </CardHeader>
      
      <CardContent className="p-0 flex flex-col h-full">
        <div className="border-t flex-1 flex flex-col overflow-hidden">
          {loading && (
            <div className="flex items-center justify-center h-96">
              <div className="text-center text-gray-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p>Loading PDF...</p>
              </div>
            </div>
          )}
          
          <div 
            className="flex-1 bg-gray-100 overflow-y-auto overflow-x-hidden" 
            style={{ 
              maxHeight: '100%',
              scrollPaddingBottom: '20px'
            }}
          >
            <div className="flex justify-center p-4 pb-12 min-h-full w-full">
              <Document
                file={file}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center text-gray-500">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                      <p>Loading PDF...</p>
                    </div>
                  </div>
                }
                error={
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center text-red-500">
                      <FileText className="w-16 h-16 mx-auto mb-4 text-red-300" />
                      <p className="text-lg font-medium">Failed to load PDF</p>
                      <p className="text-sm">Please check the file and try again</p>
                    </div>
                  </div>
                }
                noData={
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center text-gray-500">
                      <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                      <p className="text-lg font-medium">No PDF data</p>
                      <p className="text-sm">Please select a valid PDF file</p>
                    </div>
                  </div>
                }
              >
                <div className="inline-block mb-4 max-w-full">
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    rotate={rotation}
                    onLoadError={onPageLoadError}
                    onRenderSuccess={onPageRenderSuccess}
                    width={containerWidth > 0 ? containerWidth - 32 : undefined}
                    loading={
                      <div className="flex items-center justify-center h-96">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                      </div>
                    }
                    error={
                      <div className="flex items-center justify-center h-96 text-red-500">
                        <p>Failed to load page {pageNumber}</p>
                      </div>
                    }
                    className="shadow-lg block max-w-full h-auto"
                  />
                  {/* Visual indicator for bottom of content */}
                  <div className="text-center text-xs text-gray-400 mt-2 py-2">
                    Page {pageNumber} of {numPages} • Scale: {Math.round(scale * 100)}%
                  </div>
                </div>
              </Document>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 