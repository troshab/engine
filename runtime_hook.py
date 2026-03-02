import os
import shutil
import sys

_meipass = getattr(sys, '_MEIPASS', None)
if _meipass:
    _bundled = os.path.join(_meipass, 'deeprhythm-0.7.pth')
    if os.path.isfile(_bundled):
        _cache_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', 'deeprhythm')
        _cache_path = os.path.join(_cache_dir, 'deeprhythm-0.7.pth')
        if not os.path.isfile(_cache_path):
            os.makedirs(_cache_dir, exist_ok=True)
            shutil.copy2(_bundled, _cache_path)
