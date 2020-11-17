import profile, sys

if sys.argv[-1] != 'skip':
    try:
        profile.run('import lj_test; lj_test.run_sim()', 'profile.tmp')
    except:
        pass

import pstats
p = pstats.Stats('profile.tmp')
p.sort_stats('cumulative').print_stats(25)
