import matplotlib.font_manager as fm, matplotlib.pyplot as plt

print([f.name for f in fm.fontManager.ttflist if "Domine" in f.name])
# Expect something like ['Domine', 'Domine Bold', …]

plt.rcParams["font.family"] = "Domine"