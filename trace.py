





import matplotlib.pyplot as plt
import seaborn as sns
import sys
from multi_agent_kinetics import viz, serialize



filepath = sys.argv[1]
world = serialize.load_world(filepath)[0]
fig, ax = viz.set_up_figure()
viz.trace_trajectories(world, fig, ax, 'Trace')
plt.show(block=True)