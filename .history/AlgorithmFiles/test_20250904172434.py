import matplotlib.font_manager as fm
import matplotlib
print(matplotlib.matplotlib_fname())
print([f.name for f in fm.fontManager.ttflist if "Domine" in f.name])
