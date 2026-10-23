import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Les modules du projet font tous `os.chdir`-independant mais lisent des
# chemins relatifs ("data/processed", ...) : on se place à la racine du
# projet pour que ces chemins relatifs résolvent correctement en test.
os.chdir(PROJECT_ROOT)
