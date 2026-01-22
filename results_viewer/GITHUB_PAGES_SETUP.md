# GitHub Pages Setup Guide

This guide explains how to host the results viewer on GitHub Pages.

## Option 1: Serve from `results_viewer` subdirectory (Recommended)

If your repository is `username/evalscope-experiment-runner`, the website will be available at:
```
https://username.github.io/evalscope-experiment-runner/results_viewer/
```

### Steps:

1. **Ensure your files are committed:**
   ```bash
   git add results_viewer/index.html
   git add experiment_results/results.json
   git commit -m "Add results viewer for GitHub Pages"
   git push
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages** (in the left sidebar)
   - Under **Source**, select **Deploy from a branch**
   - Choose **main** (or your default branch) and **/ (root)**
   - Click **Save**

3. **Wait for deployment:**
   - GitHub will build and deploy your site (usually takes 1-2 minutes)
   - You'll see a green checkmark when it's ready
   - Your site will be live at the URL shown in the Pages settings

4. **Access your site:**
   - Visit: `https://username.github.io/evalscope-experiment-runner/results_viewer/`
   - The viewer will automatically try to load `results.json` from `../experiment_results/results.json`

## Option 2: Serve from repository root

If you want the site at `https://username.github.io/evalscope-experiment-runner/`:

1. **Move files to root:**
   ```bash
   cp results_viewer/index.html ./index.html
   ```

2. **Update the fetch path in `index.html`:**
   Change `../experiment_results/results.json` to `experiment_results/results.json`

3. **Enable GitHub Pages** (same as Option 1, step 2)

## Option 3: Use `docs` folder

1. **Create a `docs` folder and copy files:**
   ```bash
   mkdir -p docs
   cp results_viewer/index.html docs/
   # Keep experiment_results at root or copy it
   ```

2. **Update paths in `index.html`** to point to the correct location

3. **Enable GitHub Pages:**
   - In Settings → Pages, select **Deploy from a branch**
   - Choose **main** and **/docs**

## File Size Considerations

GitHub has a 100MB file size limit. If `results.json` is very large:
- Consider compressing it or splitting it
- Or use the file picker feature in the viewer to let users upload their own file
- Or host the JSON file elsewhere (e.g., GitHub Releases, raw.githubusercontent.com)

## Custom Domain (Optional)

You can use a custom domain:
1. Add a `CNAME` file in your repository root with your domain name
2. Configure DNS records as instructed by GitHub

## Troubleshooting

- **404 errors**: Check that file paths are correct relative to your Pages URL
- **JSON not loading**: Use browser DevTools (F12) → Network tab to see fetch errors
- **Build failures**: Check the Actions tab for deployment logs
- **Large files**: If `results.json` is > 50MB, consider using the file picker instead
