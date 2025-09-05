import matplotlib.font_manager as fm
print([f.name for f in fm.fontManager.ttflist if "Domine" in f.name])
