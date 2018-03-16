from astropy.table import Table
from abundances_analysis import *

selected_source_ids = [
2885707471559871872,4739429409647065856,4754387681226974208,4764349531452773120,4764753361457468672,4789117783214391680,4800105546508436736,4929932787139479552,4950236197059877632,4965577717161402880,5059628017656006784
]

data_dir = '/home/klemen/data4_mount/'
match_data = Table.read(data_dir + 'galah_tgas_xmatch_20180222.csv')

idx_use = np.in1d(match_data['source_id'], selected_source_ids)

if np.sum(idx_use) > 0:
    selected_sobject_ids = match_data['sobject_id'][idx_use]
    cannon_data = Table.read(data_dir + 'sobject_iraf_iDR2_180108_cannon.fits')
    cannon_data_sel = cannon_data[np.in1d(cannon_data['sobject_id'], selected_sobject_ids)]
    plot_abundances_histograms(cannon_data_sel, other_data=None, use_flag=True, path='test.png')