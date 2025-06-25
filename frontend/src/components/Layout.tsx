import type { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
  className?: string
}

export default function Layout({ children, className = '' }: LayoutProps) {
  return (
    <div className={`min-h-screen bg-background text-foreground ${className}`}>
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
} 