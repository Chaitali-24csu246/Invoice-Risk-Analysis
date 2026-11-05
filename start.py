"""First run installs packages into a project-local environment, then opens the app."""
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser

root = Path(__file__).resolve().parent
os.chdir(root)
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit('Please install Python 3.13 (recommended), 3.12 or 3.11 and try again.')
folder = root / '.venv'
python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not python.exists():
    print('Preparing the app environment…')
    venv.EnvBuilder(with_pip=True).create(folder)
requirements = root / 'requirements.txt'
signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
marker = folder / 'invoice-installed.txt'
if not marker.exists() or marker.read_text() != signature:
    print('Installing dependencies. Internet is needed on the first run only.')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    marker.write_text(signature)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
url = f'http://127.0.0.1:{port}'
def open_when_ready():
    for _ in range(60):
        try:
            urllib.request.urlopen(url + '/_stcore/health', timeout=1).close()
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)
threading.Thread(target=open_when_ready, daemon=True).start()
print(f'Invoice Lens: {url}\nKeep this window open. Press Ctrl+C to stop.')
try:
    subprocess.run([str(python), '-m', 'streamlit', 'run', 'app.py', '--server.address=127.0.0.1', f'--server.port={port}'], check=True)
except KeyboardInterrupt:
    pass
