import os
from plotting.figures_for_paper import FigurePlot

F = FigurePlot()

# F.Figure1()
# F.Figure2()
# F.Figure3()
# F.Figure4()
F.Figure5()
F.Figure6()
# F.NEW_FIGURES()

os.system('pdfunite results/figures/**/*.pdf results/figures/all_figures.pdf')
