# AI Document Assistant - Frontend

A modern React frontend built with Vite, TypeScript, Tailwind CSS, and shadcn/ui for the AI Document Assistant application.

## Features

- **Modern Stack**: Vite + React 18 + TypeScript
- **Styling**: Tailwind CSS with shadcn/ui components
- **Icons**: Lucide React icons
- **Responsive Design**: Mobile-first responsive layout
- **Dark Mode Ready**: Built-in dark mode support with shadcn/ui

## Project Structure

```
src/
├── components/          # Reusable UI components
│   └── ui/             # shadcn/ui components
├── pages/              # Page components
├── lib/                # Utility functions
├── hooks/              # Custom React hooks
└── App.tsx             # Main app component
```

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Components

### UI Components (shadcn/ui)
- `Button` - Various button styles and sizes
- `Card` - Card layouts with header, content, and footer
- More components can be added as needed

### Pages
- `Home` - Main landing page with upload and question features

## Styling

The project uses Tailwind CSS with a custom theme configured for shadcn/ui:
- CSS variables for theming
- Dark mode support
- Custom color palette
- Responsive utilities

## Development

### Adding New shadcn/ui Components
To add new shadcn/ui components, you can manually create them following the shadcn/ui patterns or use the CLI (when available for Vite projects).

### Path Aliases
The project is configured with path aliases:
- `@/` maps to `src/`
- Example: `import { Button } from "@/components/ui/button"`

## API Integration
The frontend is designed to integrate with the FastAPI backend for:
- PDF upload and processing
- Document question answering
- RAG pipeline interactions

## Environment Variables
Create a `.env` file for environment-specific configuration:
```
VITE_API_URL=http://localhost:8000
```

## Browser Support
- Modern browsers with ES2022 support
- Chrome 87+
- Firefox 78+
- Safari 14+
- Edge 88+
