# Velocity humanize API

CPU FastAPI service for the piano velocity diffusion model.

**Deploy:** [Render](https://render.com) → New Web Service → this repo → Docker. See `render.yaml`.

After deploy, set `window.HUMANIZE_API` in [krishivsinghal.github.io `piano/config.js`](https://github.com/krishivsinghal/krishivsinghal.github.io/blob/main/piano/config.js) to your service URL (no trailing slash).

- `GET /health`
- `POST /humanize` — multipart `.mid` / `.midi`
