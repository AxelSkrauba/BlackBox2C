"""MkDocs hook: copy notebooks from notebooks/ into docs/notebooks/ before build."""
import os
import shutil


def on_pre_build(config):
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")
    dst = os.path.join(os.path.dirname(__file__), "notebooks")
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
