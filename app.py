import os
import sys

if os.environ.get("TIMEMACHINE_DEV_RELOAD") == "1":
    import streamlit as st

    for key in ["spec", "run_result", "adapter"]:
        if hasattr(st, "session_state") and key in st.session_state:
            del st.session_state[key]
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("timemachine"):
            del sys.modules[mod_name]

from timemachine.ui.app_view import render_app

if __name__ == "__main__":
    render_app()
else:
    render_app()

