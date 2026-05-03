# Setting up a local reverse proxy

## Starting
From the reverse proxy directory. Make sure cloudflared and python route their requests through a VPN!

```ps1
cloudflared tunnel --url http://127.0.0.1:5001
.\.venv\Scripts\Activate.ps1
python server.py
```
