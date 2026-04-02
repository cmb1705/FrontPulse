from pathlib import Path

patterns = ['phase0_', 'phase1_', 'phase2_', 'phase3_', 'phase4_', 'phase5_', 'phase2 ', 'phase3 ', 'phase4 ', 'phase5 ']
skip_dirs = {'data','docs','references','.git','.venv','__pycache__'}
exts = {'.py','.md','.txt','.yaml','.yml','.json','.rst','.cfg','.ini'}
root = Path('.')
for pattern in patterns:
    print(f"--- searching for {pattern} ---")
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in exts:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        if pattern in text:
            print(path)
