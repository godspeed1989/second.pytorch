import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import sys

linestyles = OrderedDict(
                         [('solid',               (0, ())),
                          ('loosely dotted',      (0, (1, 10))),
                          ('dotted',              (0, (1, 5))),
                          ('densely dotted',      (0, (1, 1))),
                          ('loosely dashed',      (0, (5, 10))),
                          ('dashed',              (0, (5, 5))),
                          ('densely dashed',      (0, (5, 1))),
                          ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                          ('dashdotted',          (0, (3, 5, 1, 5))),
                          ('densely dashdotted',  (0, (3, 1, 1, 1))),
                          ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                          ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                          ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

ROOT_DIR = sys.argv[1]

det_3d = [ [] for _ in range(3) ]
det_bv = [ [] for _ in range(3) ]

log_file = os.path.join(ROOT_DIR, 'log.txt')
if not os.path.exists( log_file ):
    print('log file not exist')
    sys.exit(0)
else:
    lines = open(log_file).readlines()
    cnt_3d = 0
    cnt_bev = 0
    for line in lines:
        line = line.split()
        if len(line) and line[0] == 'bev':
            if cnt_3d % 3 == 0:
                det_bv[0].append( float( line[-3][3:-1] ) )
                det_bv[1].append( float( line[-2][:-1] ) )
                det_bv[2].append( float( line[-1][:-1] ) )
            cnt_3d += 1
        if len(line) and line[0] == '3d':
            if cnt_bev % 3 == 0:
                det_3d[0].append( float(line[-3][3:-1]) )
                det_3d[1].append( float(line[-2][:-1]) )
                det_3d[2].append( float(line[-1][:-1]) )
            cnt_bev += 1

RANGE = range(len(det_bv[0]))
print('num epoch', len(det_bv[0]))

plt.figure(figsize=(10, 7))

plt.plot( RANGE,  det_3d[0] , linestyle=linestyles['solid'], linewidth=1.5, color=tableau20[0] )
plt.plot( RANGE,  det_3d[1] , linestyle=linestyles['solid'], linewidth=1.5, color=tableau20[2] )
plt.plot( RANGE,  det_3d[2] , linestyle=linestyles['solid'], linewidth=1.5, color=tableau20[4] )
plt.plot( RANGE,  det_bv[0] , linestyle=linestyles['densely dotted'], linewidth=1.5, color=tableau20[0] )
plt.plot( RANGE,  det_bv[1] , linestyle=linestyles['densely dotted'], linewidth=1.5, color=tableau20[2] )
plt.plot( RANGE,  det_bv[2] , linestyle=linestyles['densely dotted'], linewidth=1.5, color=tableau20[4] )

plt.legend(['3d easy', '3d moderate', '3d hard', 'bird view easy', 'bird view moderate', 'bird view hard'], loc=4)

plt.xlabel('Epoch',  fontsize=16)
plt.xticks(  range(1, len(det_bv[0]), 2)  )
plt.xticks(fontsize=14)

plt.tick_params(labelright=True)
plt.ylabel('AP', fontsize=16)
plt.yticks( range(0, 100, 5) )
plt.yticks(fontsize=14)

plt.grid(linestyle=linestyles['dotted'])

out_name = os.path.dirname(ROOT_DIR)
OUTPUT_NAME = out_name + '.png'
plt.savefig(OUTPUT_NAME)

print('results parsed and saved in: ' + OUTPUT_NAME)
