import sys

# Force reload of timemachine package on Streamlit rerun so code edits take effect immediately
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("timemachine"):
        del sys.modules[mod_name]

from timemachine.ui.app_view import render_app

if __name__ == "__main__":
    render_app()
else:
    render_app()
