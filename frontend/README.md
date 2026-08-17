# Frontend — Autonomous Multi-Agent Research Assistant

Next.js 16 (App Router) UI for the research pipeline. See the [root README](../README.md) for the full project, architecture, and deployment notes.

## Develop

```bash
npm install
cp .env.example .env.local     # NEXT_PUBLIC_BACKEND_URL, defaults to http://localhost:8000
npm run dev                    # http://localhost:3000
```

The backend must be running separately (`cd ../backend && uvicorn app.main:app --reload --port 8000`). With the backend down, every page shows an error naming the URL it tried — it does **not** fall back to sample data.

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run build
```

## Build output

`next.config.ts` sets `output: "export"`, so `npm run build` emits a **static** bundle to `out/`. There is no SSR and no Node server at runtime: every route is prerendered HTML, and all backend communication happens from the browser.

Two consequences worth knowing:

- `NEXT_PUBLIC_BACKEND_URL` is inlined at build time. Changing it requires a rebuild.
- Anything needing a request-time server (middleware, route handlers, ISR) is unavailable by design.

## Routes

| Route | Purpose |
|---|---|
| `/` | Prompt, live WebSocket progress, results, report downloads |
| `/history` | Prior runs from `GET /history` |
| `/reports` | Latest report plus `.md` / `.json` / `.pdf` downloads |
| `/reports/detail?id=<run_id>` | One run's full report |
| `/dashboard` | Run counts and critic-score stats, derived from stored history |

## Deploy (Firebase Hosting)

```bash
npm run build
firebase deploy --only hosting
```

`firebase.json` serves `out/` and rewrites all paths to `/index.html`.
