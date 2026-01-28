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

### Access from Internal Network

To allow others on your network to access the viewer:

**Option 1: Use the helper script (easiest)**
```bash
cd results_viewer
./serve_local.sh
```
This script will automatically find your IP and start the server. It will show you both local and network URLs.

**Option 2: Manual setup**

1. **Start the server bound to your network interface:**
   ```bash
   # From repository root
   python -m http.server 8000 --bind 0.0.0.0
   # or
   python3 -m http.server 8000 --bind 0.0.0.0
   ```

2. **Find your local IP address:**
   ```bash
   # Linux
   hostname -I
   # or
   ip addr show | grep "inet " | grep -v 127.0.0.1
   
   # Mac
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

3. **Access from other devices on the same network:**
   ```
   http://YOUR_LOCAL_IP:8000/results_viewer/
   # Example: http://192.168.1.100:8000/results_viewer/
   ```

4. **Make sure firewall allows connections:**
   ```bash
   # Linux (if using ufw)
   sudo ufw allow 8000/tcp
   
   # Check firewall status
   sudo ufw status
   ```

### GitHub Pages (Public Access)

Your GitHub Pages site is accessible from anywhere with internet, including your internal network:

**Your site URL:**
```
https://eyuansu62.github.io/flageval-1m/results_viewer/
```

**To access from your network:**
- Simply open the URL above in any browser on any device connected to the internet
- No special configuration needed - GitHub Pages is publicly accessible
- Works on phones, tablets, other computers, etc.

See [GITHUB_PAGES_SETUP.md](./GITHUB_PAGES_SETUP.md) for detailed setup instructions.

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
