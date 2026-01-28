# GitHub Pages Setup Guide

This guide explains how to host the results viewer on GitHub Pages.

## ⚠️ Important: `experiment_results/` is gitignored

Since `experiment_results/` is in `.gitignore`, you need to copy `results.json` to a location that will be committed. Choose one of the options below.

## Option 1: Copy results.json to results_viewer (Recommended)

This is the simplest approach for your repository `eyuansu62/flageval-1m`.

### Steps:

1. **Copy results.json to the viewer directory:**
   ```bash
   cp experiment_results/results.json results_viewer/results.json
   ```

2. **Update the HTML to look for the local file first:**
   The HTML already tries multiple paths, but we should prioritize the local one. The current code will work, but you can also manually update the fetch paths if needed.

3. **Commit and push:**
   ```bash
   git add results_viewer/index.html
   git add results_viewer/results.json
   git commit -m "Add results viewer for GitHub Pages"
   git push
   ```

4. **Enable GitHub Pages (if not already enabled):**
   - Go to your repository on GitHub: `https://github.com/eyuansu62/flageval-1m`
   - Click **Settings** → **Pages** (in the left sidebar)
   - Under **Source**, select **Deploy from a branch**
   - Choose **main** (or your default branch) and **/ (root)**
   - Click **Save**

5. **Access your site:**
   - Your site will be at: `https://eyuansu62.github.io/flageval-1m/results_viewer/`
   - The viewer will automatically load `results.json` from the same directory

## Option 2: Use GitHub Releases or Raw GitHub URLs

If `results.json` is too large or you don't want to commit it:

1. **Upload results.json to GitHub Releases:**
   - Go to your repo → Releases → Create a new release
   - Upload `results.json` as an asset
   - Get the direct download URL

2. **Or use raw.githubusercontent.com:**
   - Commit `results.json` to a branch (even if gitignored, you can force add it)
   - Use URL: `https://raw.githubusercontent.com/eyuansu62/flageval-1m/main/path/to/results.json`

3. **Update the HTML fetch path** to use the direct URL

## Option 3: Serve from repository root

If you want the site at `https://eyuansu62.github.io/flageval-1m/`:

1. **Copy files to root:**
   ```bash
   cp results_viewer/index.html ./index.html
   cp experiment_results/results.json ./results.json
   ```

2. **Commit and push:**
   ```bash
   git add index.html results.json
   git commit -m "Add results viewer to root"
   git push
   ```

3. **Enable GitHub Pages** (same as Option 1, step 4)

## Option 4: Use `docs` folder

1. **Create a `docs` folder and copy files:**
   ```bash
   mkdir -p docs
   cp results_viewer/index.html docs/
   cp experiment_results/results.json docs/results.json
   ```

2. **Commit and push:**
   ```bash
   git add docs/
   git commit -m "Add results viewer to docs folder"
   git push
   ```

3. **Enable GitHub Pages:**
   - In Settings → Pages, select **Deploy from a branch**
   - Choose **main** and **/docs**
   - Your site will be at: `https://eyuansu62.github.io/flageval-1m/`

## Quick Setup Summary

**For your repository (`eyuansu62/flageval-1m`):**

```bash
# 1. Copy results.json to results_viewer directory
cp experiment_results/results.json results_viewer/results.json

# 2. Commit and push
git add results_viewer/
git commit -m "Add results viewer"
git push

# 3. Your site will be at:
# https://eyuansu62.github.io/flageval-1m/results_viewer/
```

The HTML already tries to load `results.json` from the same directory first, so it will work automatically!

## File Size Considerations

GitHub has a 100MB file size limit per file. If `results.json` is very large:
- Consider compressing it or splitting it
- Or use the file picker feature in the viewer (users can upload their own file)
- Or host the JSON file elsewhere (e.g., GitHub Releases, raw.githubusercontent.com)
- Or use Git LFS (Large File Storage) if the file is between 50-100MB

## Custom Domain (Optional)

You can use a custom domain:
1. Add a `CNAME` file in your repository root with your domain name
2. Configure DNS records as instructed by GitHub

## Troubleshooting

- **404 errors**: Check that file paths are correct relative to your Pages URL
- **JSON not loading**: Use browser DevTools (F12) → Network tab to see fetch errors
- **Build failures**: Check the Actions tab for deployment logs
- **Large files**: If `results.json` is > 50MB, consider using the file picker instead
