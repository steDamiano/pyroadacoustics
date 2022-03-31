import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "pgf.texsystem":   "pdflatex", # or any other engine you want to use
        "text.usetex":     True,       # use TeX for all texts
        "font.family":     "serif",
        "font.serif":      [],         # empty entries should cause the usage of the document fonts
        "font.sans-serif": [],
        "font.monospace":  [],
        "font.size":       10,         # control font sizes of different elements
        "axes.labelsize":  10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pgf.preamble": [              # specify additional preamble calls for LaTeX's run
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{siunitx}",
        ],
    }
)
import matplotlib.pyplot as plt
from scipy.stats import norm
x = norm.rvs(size = 100)
y = norm.pdf(x)

plt.scatter(x,y,marker = 'x')
plt.savefig('norm.pdf', format = 'pdf')
