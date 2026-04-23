# SMB Ad Manager — Frontend

Next.js 15 + Tailwind + Framer Motion front end for the reward-hardened SMB Ad Manager RL environment.

Hosted backend API: https://falgunisharma-smb-ad-manager.hf.space
Backend source: `../` (this repo's root)

## Phase 1 pages

- `/login` — cookie-based auth gate (demo creds: `admin / hackathon2026`)
- `/dashboard` — project overview, live env health check, stat cards
- `/playground` — interactive agent demo against the live HF Space environment

Coming in Phase 2:
- `/adversarial` — reward-hacking detection live demo
- `/metrics` — training curves + before/after
- `/about` — team + tech + citations

## Local dev

```bash
cd frontend
cp .env.example .env.local   # then edit .env.local if you want to change the demo password
npm install
npm run dev                   # http://localhost:3000
```

## Deploy to Vercel

```bash
# first time only:
npx vercel login
npx vercel link             # point this directory at a new Vercel project

# or use the Vercel dashboard:
# 1. https://vercel.com/new
# 2. Import the Falgunisharma72/smb-ad-manager repo
# 3. Set root directory to "frontend"
# 4. Set env vars (copy from .env.example):
#      NEXT_PUBLIC_API_URL=https://falgunisharma-smb-ad-manager.hf.space
#      AUTH_USERNAME=admin
#      AUTH_PASSWORD=<your-demo-password>
# 5. Deploy.
```

## Stack

- **Next.js 15** App Router
- **TypeScript** (strict)
- **Tailwind CSS** + custom pastel theme (cream / sage / peach / lavender)
- **Framer Motion** for page transitions, hover effects, staggered reveals
- **lucide-react** icons
- **clsx + tailwind-merge** for conditional classes

## Theme tokens

Defined in `app/globals.css` as CSS variables, consumed by `tailwind.config.ts`:

| Token | Hex (approx) | Use |
|---|---|---|
| `background` | `#FFFBF5` | page bg |
| `foreground` | `#2D3748` | body text |
| `primary` | `#9FB89A` | sage — main accent |
| `accent-peach` | `#F4BFAA` | |
| `accent-lavender` | `#C8B5E0` | |
| `accent-rose` | `#EFC6C2` | |
| `accent-sky` | `#B8D4E5` | |

## Animations

- Page-level `FadeIn` / `StaggerChildren` components for staged reveals
- Hover lift on cards + buttons
- Floating hero blobs (gradient + transform loop)
- Animated number counters on stat cards
- Shared `layoutId` transitions on active nav link
- Fade-in/out on route change in the playground
