# Experiment Results Viewer

A simple, self-contained web viewer for experiment results.

## Quick Start

### Local Development

1. Start a local server from the repository root:
   ```bash
   python -m http.server 8000
   # or
   python3 -m http.server 8000
   ```

2. Open in browser:
   ```
   http://localhost:8000/results_viewer/
   ```

### GitHub Pages

See [GITHUB_PAGES_SETUP.md](./GITHUB_PAGES_SETUP.md) for detailed instructions.

**Quick setup:**
1. Commit and push your files
2. Go to repository Settings → Pages
3. Select "Deploy from a branch" → choose `main` and `/ (root)`
4. Your site will be at: `https://username.github.io/repo-name/results_viewer/`

## Features

- 📊 Summary statistics dashboard
- 🔍 Filter by status, search models, filter by score
- 📈 Interactive charts and visualizations
- 📋 Detailed model information
- 💾 File picker fallback if auto-load fails

## File Structure

```
results_viewer/
├── index.html              # Main viewer (self-contained)
├── README.md              # This file
└── GITHUB_PAGES_SETUP.md  # GitHub Pages setup guide
```

The viewer automatically tries to load `../experiment_results/results.json` and will try multiple paths to work with different GitHub Pages configurations.
