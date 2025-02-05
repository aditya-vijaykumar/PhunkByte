# PhunkByte

A decentralized application built with Next.js, Express.js, and Noir smart contracts.

## Project Structure

```
packages/
├── frontend/     # Next.js frontend application
├── backend/      # Express.js backend server
└── contracts/    # Noir smart contracts
```

## Prerequisites

- Node.js (v18 or later)
- pnpm (`npm install -g pnpm`)
- Nargo (for Noir contracts)

## Getting Started

1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Start development servers:
   ```bash
   # Start all services in parallel
   pnpm dev

   # Or start individually
   pnpm frontend  # Start Next.js frontend
   pnpm backend   # Start Express backend
   ```

## Development

- Frontend runs on: http://localhost:3000
- Backend runs on: http://localhost:4000

## License

MIT