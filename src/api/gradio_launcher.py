"""
Compatibility wrapper: run the Gradio UI from a separate process.

The project contains `gradio_app.py` which imports the installed `gradio` package.
This small launcher avoids circular imports that can occur when the module name
`gradio` collides with the package name.

Usage (from project `src/` directory):
    python api/gradio.py

This will spawn a subprocess running `gradio_app.py`.
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    target = os.path.join(this_dir, "gradio_app.py")
    if not os.path.exists(target):
        print("gradio_app.py not found. Please ensure src/api/gradio_app.py exists.")
        sys.exit(1)
    # Launch the gradio UI in a separate process to avoid import name collisions
    subprocess.run([sys.executable, target])
