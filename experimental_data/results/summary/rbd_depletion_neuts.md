# Determine the amount of neutralizing activity directed towards the RBD elicited by infection with Delta variant

## Set up analysis
Import packages


```python
import os
import re
import warnings

from IPython.display import display, HTML

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort

import numpy as np
import pandas as pd
from plotnine import *
import seaborn

import neutcurve
from neutcurve.colorschemes import CBMARKERS, CBPALETTE
import seaborn

import lifelines
import sklearn
import scipy.stats
from statistics import geometric_mean

import yaml

print(f"Using `neutcurve` version {neutcurve.__version__}")
```

    Using `neutcurve` version 0.5.7


Use seaborn theme and change font:


```python
theme_set(theme_seaborn(style='white', context='talk', font='FreeSans', font_scale=1))
plt.style.use('seaborn-white')
```

Read in config file


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Define results directory


```python
resultsdir = 'results/rbd_depletion_neuts'
os.makedirs(resultsdir, exist_ok=True)
```

## Read in fracinfect file and plot curves

Read in the neut data for all experiments for this paper, but only select the date(s) that correspond to the RBD depletions experiment. Note that this is only configured to handle one experiment date currently (because the amazingly talented Rachel Eguia ran all these assays on a single date). So for future projects, this might need to be adjusted. 


```python
rbd_depletions_date = [str(i) for i in config['rbd_depletions_date']]
heterologous_depletions_dates = [str(i) for i in config['heterologous_depletions_dates']]
print(f'Getting data from {rbd_depletions_date}')

frac_infect = (pd.read_csv(config['aggregate_fract_infect_csvs'])
               .query('date in @rbd_depletions_date')
               .assign(virus=lambda x: np.where((x['virus']=="mock depletion (D614G spike)") & (x['date'].isin(heterologous_depletions_dates)), 
                                                'heterologous mock depletion (D614G spike)',
                                                x['virus']
                                               )
                      )
               .assign(virus=lambda x: np.where((x['virus']=="Delta RBD antibodies depleted x D614G spike PV"), 
                                                'heterologous Ab depletion (D614G spike)',
                                                x['virus']
                                               )
                      )
              )

fits = neutcurve.CurveFits(frac_infect)

fitparams = (
    fits.fitParams()
    .assign(spike=lambda x: np.where(x['virus'].str.contains('D614G'), 'D614G', 'Delta'))
    .assign(spike=lambda x: np.where(x['virus'].str.contains('heterologous'), 'Delta dep x D614G PV', x['spike']))
    # get columns of interest
    [['serum', 'virus', 'ic50', 'ic50_bound', 'spike']]
    .assign(NT50=lambda x: 1/x['ic50'])
    .assign(depletion=lambda x: np.where(x['virus'].str.contains('mock'), 'mock depletion', 'RBD antibodies depleted'))
    )

# couldn't get lambda / conditional statement to work with assign, so try it here:
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

display(HTML(fitparams.head().to_html(index=False)))
```

    Getting data from ['2021-11-12', '2021-11-25', '2021-12-12', '2021-12-22', '2022-01-06']


    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/neutcurve/hillcurve.py:741: RuntimeWarning: invalid value encountered in power
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/scipy/optimize/minpack.py:833: OptimizeWarning: Covariance of the parameters could not be estimated
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/neutcurve/hillcurve.py:451: RuntimeWarning: invalid value encountered in sqrt



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>virus</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>spike</th>
      <th>NT50</th>
      <th>depletion</th>
      <th>ic50_is_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Delta_10</td>
      <td>heterologous mock depletion (D614G spike)</td>
      <td>0.001134</td>
      <td>interpolated</td>
      <td>Delta dep x D614G PV</td>
      <td>882.157389</td>
      <td>mock depletion</td>
      <td>False</td>
    </tr>
    <tr>
      <td>Delta_10</td>
      <td>heterologous Ab depletion (D614G spike)</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>Delta dep x D614G PV</td>
      <td>25.000000</td>
      <td>RBD antibodies depleted</td>
      <td>True</td>
    </tr>
    <tr>
      <td>Delta_10</td>
      <td>mock depletion (D614G spike)</td>
      <td>0.001088</td>
      <td>interpolated</td>
      <td>D614G</td>
      <td>918.856380</td>
      <td>mock depletion</td>
      <td>False</td>
    </tr>
    <tr>
      <td>Delta_10</td>
      <td>RBD antibodies depleted (D614G spike)</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>D614G</td>
      <td>25.000000</td>
      <td>RBD antibodies depleted</td>
      <td>True</td>
    </tr>
    <tr>
      <td>Delta_10</td>
      <td>mock depletion (Delta spike)</td>
      <td>0.000322</td>
      <td>interpolated</td>
      <td>Delta</td>
      <td>3102.893168</td>
      <td>mock depletion</td>
      <td>False</td>
    </tr>
  </tbody>
</table>



```python
non_neut=[]

fig, axes = fits.plotSera(sera=natsort.natsorted(fitparams.query('serum not in @non_neut')['serum'].unique()),
                          xlabel='serum dilution',
                          ncol=4,
                          widthscale=1,
                          heightscale=1,
                          titlesize=20, labelsize=24, ticksize=15, legendfontsize=20, yticklocs=[0,0.5,1],
                          markersize=8, linewidth=2,
                          max_viruses_per_subplot=8,
                          fix_lims={'ymax':1.25},
                         )

plotfile = PdfPages(f'{resultsdir}/sera_frac_infectivity.pdf')
plotfile.savefig(bbox_inches='tight', transparent=True)
plotfile.close()
```


    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_10_0.png)
    



```python
# define which sera to exclude from downstream analyses
exclude_sera = []
```


```python
sample_key = pd.read_csv(config['sample_key_file'])

foldchange = (
    fitparams
    .query('serum not in @exclude_sera')
    .pivot_table(values='ic50', index=['serum','spike'], columns=['depletion'])
    .reset_index()
    .rename(columns={'RBD antibodies depleted': 'post-depletion_ic50', 'mock depletion': 'pre-depletion_ic50'})
    .assign(fold_change=lambda x: x['post-depletion_ic50'] / x['pre-depletion_ic50'],
            percent_RBD= lambda x: ((1-1/x['fold_change'])*100).astype(int),
            NT50_pre=lambda x: 1/x['pre-depletion_ic50'],
            NT50_post=lambda x: 1/x['post-depletion_ic50'],
           )
    .merge((fitparams
            .query('depletion=="RBD antibodies depleted"')
            [['serum', 'spike','ic50_is_bound']]
            .drop_duplicates()
           ), 
           on=['serum','spike'],
           how='left',
          )
    .assign(perc_RBD_str = lambda x: x['percent_RBD'].astype(str)
           )
    .rename(columns={'ic50_is_bound': 'post_ic50_bound'})
    .merge(fitparams)
    .assign(depletion=lambda x: pd.Categorical(x['depletion'], categories=['mock depletion', 'RBD antibodies depleted'], ordered=True))
    .merge(sample_key[['subject_name', 'sample_type']].rename(columns={'subject_name':'serum'}), 
           on='serum'
          )
    )

foldchange['perc_RBD_str'] = np.where(foldchange['post_ic50_bound'], '>'+foldchange['perc_RBD_str']+'%', foldchange['perc_RBD_str']+'%')
display(HTML(foldchange.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>spike</th>
      <th>post-depletion_ic50</th>
      <th>pre-depletion_ic50</th>
      <th>fold_change</th>
      <th>percent_RBD</th>
      <th>NT50_pre</th>
      <th>NT50_post</th>
      <th>post_ic50_bound</th>
      <th>perc_RBD_str</th>
      <th>virus</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>NT50</th>
      <th>depletion</th>
      <th>ic50_is_bound</th>
      <th>sample_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C</td>
      <td>D614G</td>
      <td>0.040000</td>
      <td>0.000645</td>
      <td>62.002654</td>
      <td>98</td>
      <td>1550.066341</td>
      <td>25.00000</td>
      <td>True</td>
      <td>&gt;98%</td>
      <td>mock depletion (D614G spike)</td>
      <td>0.000645</td>
      <td>interpolated</td>
      <td>1550.066341</td>
      <td>mock depletion</td>
      <td>False</td>
      <td>Delta breakthrough</td>
    </tr>
    <tr>
      <td>267C</td>
      <td>D614G</td>
      <td>0.040000</td>
      <td>0.000645</td>
      <td>62.002654</td>
      <td>98</td>
      <td>1550.066341</td>
      <td>25.00000</td>
      <td>True</td>
      <td>&gt;98%</td>
      <td>RBD antibodies depleted (D614G spike)</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>RBD antibodies depleted</td>
      <td>True</td>
      <td>Delta breakthrough</td>
    </tr>
    <tr>
      <td>267C</td>
      <td>Delta</td>
      <td>0.040000</td>
      <td>0.001664</td>
      <td>24.043240</td>
      <td>95</td>
      <td>601.081003</td>
      <td>25.00000</td>
      <td>True</td>
      <td>&gt;95%</td>
      <td>mock depletion (Delta spike)</td>
      <td>0.001664</td>
      <td>interpolated</td>
      <td>601.081003</td>
      <td>mock depletion</td>
      <td>False</td>
      <td>Delta breakthrough</td>
    </tr>
    <tr>
      <td>267C</td>
      <td>Delta</td>
      <td>0.040000</td>
      <td>0.001664</td>
      <td>24.043240</td>
      <td>95</td>
      <td>601.081003</td>
      <td>25.00000</td>
      <td>True</td>
      <td>&gt;95%</td>
      <td>RBD antibodies depleted (Delta spike)</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>RBD antibodies depleted</td>
      <td>True</td>
      <td>Delta breakthrough</td>
    </tr>
    <tr>
      <td>267C</td>
      <td>Delta dep x D614G PV</td>
      <td>0.032963</td>
      <td>0.001152</td>
      <td>28.603933</td>
      <td>96</td>
      <td>867.751512</td>
      <td>30.33679</td>
      <td>False</td>
      <td>96%</td>
      <td>heterologous mock depletion (D614G spike)</td>
      <td>0.001152</td>
      <td>interpolated</td>
      <td>867.751512</td>
      <td>mock depletion</td>
      <td>False</td>
      <td>Delta breakthrough</td>
    </tr>
  </tbody>
</table>


Sort the serum names in ascending order for least to most potent against Delta spike. 


```python
serum_order = (foldchange
               [['serum', 'spike', 'NT50_pre']]
               .query('spike=="Delta"')
               .sort_values(['NT50_pre'], ascending=True)
               ['serum']
               .unique()
              )

print(len(serum_order))

assert len(serum_order) == len(foldchange['serum'].unique())
```

    24


Save `foldchange` dataframe to CSV


```python
csvfile=f'{resultsdir}/RBD_depletion_NT50.csv'
print(f'Writing to {csvfile}')
foldchange.to_csv(csvfile, index=False)
```

    Writing to results/rbd_depletion_neuts/RBD_depletion_NT50.csv


Define the ways we substitute labels for naming here: 


```python
sample_type_replacement={'2x BNT162b2': '2x\nBNT162b2',
                         'Delta breakthrough': 'Delta\nbreakthrough\nafter 2x mRNA',
                         'primary Delta infection': 'primary\nDelta infection',
                        }

dep_virus_replacement={'Delta':'RBD Abs dep: Delta\nspike PV neuts: Delta',
                        'D614G':'RBD Abs dep: D614G\nspike PV neuts: D614G',
                        'Delta dep x D614G PV':'RBD Abs dep: Delta\nspike PV neuts: D614G',
                       }

depletion_replacement={'mock depletion':'mock','RBD antibodies depleted': 'depleted'}
```


```python
p = (ggplot(foldchange
            .replace(sample_type_replacement)
            .replace(dep_virus_replacement)
            .assign(
                # options are to either order sera by name or by potency    
                # serum=lambda x: pd.Categorical(x['serum'], natsort.natsorted(x['serum'].unique())[::-1], ordered=True),
                
                # let's try potency for now
                serum=lambda x: pd.Categorical(x['serum'], serum_order, ordered=True),
                spike=lambda x: pd.Categorical (x['spike'], ordered=True, categories=dep_virus_replacement.values()), 
                sample_type=lambda x: pd.Categorical(x['sample_type'], ordered=True, categories=sample_type_replacement.values()))
            , 
            aes(x='NT50',
                y='serum',
                fill='depletion',
                group='serum',
                label='perc_RBD_str',
                # color='sample_type',
               )) +
     scale_x_log10(name='neutralization titer 50% (NT50)', 
                   limits=[config['NT50_LOD'],foldchange['NT50'].max()*3.5]) +
     geom_vline(xintercept=config['NT50_LOD'], 
                linetype='dotted', 
                size=1, 
                alpha=0.6, 
                color=CBPALETTE[7]) +
     geom_line(alpha=1, color=CBPALETTE[0]) + #
     geom_point(size=4, color=CBPALETTE[0]) + #
     geom_text(aes(x=foldchange['NT50'].max()*3.5, y='serum'), #
               color=CBPALETTE[0],
               ha='right',
               size=9,
              ) +
     theme(figure_size=(3*foldchange['spike'].nunique(),0.75+0.25*foldchange['serum'].nunique()),
           axis_text=element_text(size=12),
           legend_text=element_text(size=12),
           legend_title=element_text(size=12),
           axis_title_x=element_text(size=14),
           axis_title_y=element_text(size=14, margin={'r': 40}),
           legend_position='right',
           strip_background=element_blank(),
           strip_text=element_text(size=14),
          ) +
     facet_grid('sample_type~spike', scales='free_y' )+
     ylab('plasma') +
     scale_fill_manual(values=['#999999', '#FFFFFF', ]) #+
     # scale_color_manual(values=CBPALETTE[1:]) #+
     # labs(fill = '')
    )

_ = p.draw()

p.save(f'{resultsdir}/NT50_trackplot.pdf')
```

    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 9 x 6.75 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/NT50_trackplot.pdf



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_19_1.png)
    



```python
LOD = config['NT50_LOD']

NT50_lines = (ggplot((foldchange
                      .replace(depletion_replacement)
                      .replace(sample_type_replacement)
                      .replace(dep_virus_replacement)
                      .assign(spike=lambda x: pd.Categorical (x['spike'], 
                                                              ordered=True, 
                                                              categories=dep_virus_replacement.values()),
                              sample_type=lambda x: pd.Categorical(x['sample_type'], 
                                                                   ordered=True, 
                                                                   categories=sample_type_replacement.values()
                                                                  ),
                              depletion=lambda x: pd.Categorical(x['depletion'], 
                                                                 categories=depletion_replacement.values(), 
                                                                 ordered=True))
                     )
                     , aes(x='depletion',
                           y='NT50',
                           group='serum'
                          )
                    ) + 
              geom_point(size=2.5, alpha=0.25) +
              geom_line(alpha=0.25) +
              facet_grid('sample_type~spike') +
              theme(axis_title_y=element_text(margin={'r': 6}),
                    strip_background=element_blank(),
                    figure_size=(2*foldchange['spike'].nunique(),2*foldchange['sample_type'].nunique()),
                    axis_text_x=element_text(angle=90),
                    text=element_text(size=14),
                    strip_text_x=element_text(size=10),
                   ) +
              scale_y_log10(name='neutralization titer (NT50)') +
              xlab('mock or depletion of Delta\nRBD-binding antibodies') +
              geom_hline(yintercept=LOD,
                         color=CBPALETTE[7],
                         alpha=1,
                         size=1,
                         linetype='dotted',
                        )
             )

_ = NT50_lines.draw()
NT50_lines.save(f'{resultsdir}/compare_RBDtargeting.pdf')
```

    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 6 x 6 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/compare_RBDtargeting.pdf



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_20_1.png)
    



```python
# p = (ggplot((foldchange
#              .drop(columns=['depletion', 'NT50'])
#              .drop_duplicates()
#              .replace(dep_virus_replacement)
#              .replace(sample_type_replacement)
#              .assign(spike=lambda x: pd.Categorical (x['spike'],
#                                                      ordered=True,
#                                                      categories=dep_virus_replacement.values()))
#             )
#            ) +
#      aes('spike', 'percent_RBD') +
#      geom_boxplot(width=0.65,
#                   position=position_dodge(width=0.7),
#                   outlier_shape='') +
#      geom_jitter(position=position_dodge(width=0.7),
#                  alpha=0.25, size=2) +
#      theme(figure_size=(0.75*foldchange['spike'].nunique()*foldchange['sample_type'].nunique(),2.5),
#            strip_background=element_blank(),
#            axis_text_x=element_text(angle=90),
#            strip_text_x=element_text(size=12),
#            text=element_text(size=12),
#            ) +
#      facet_wrap('~sample_type')+
#      scale_y_continuous(limits=[70, 100]) +
#      ylab('percent neutralizing potency\ndue to RBD antibodies') +
#      xlab ('')
#      )

# _ = p.draw()
# p.save(f'{resultsdir}/compare_percentRBD.pdf')
```

Ok, this ^ is actually a pretty misleading plot, because most of the points are censored, so the percent RBD targeting is actually mostly a function of pre-depletion NT50 (i.e., more potent sera have more "room" to look like they are highly RBD-targeting before they hit the LOD on the low end).

The comparison we actually care about is how much *residual* neutralizing potency there is in each of the conditions. I wonder if I can just plot the post-depletion NT50...


```python
# p = (ggplot((foldchange
#              .query('depletion=="RBD antibodies depleted"')
#              .drop(columns=['depletion', 'NT50'])
#              .drop_duplicates()
#              .replace(dep_virus_replacement)
#              .replace(sample_type_replacement)
#              .assign(spike=lambda x: pd.Categorical (x['spike'],
#                                                      ordered=True,
#                                                      categories=dep_virus_replacement.values()))
#             )
#            ) +
#      aes('spike', 'NT50_post', group='serum') +
#      geom_line(aes(x='spike', y='NT50_post', group='serum'), color=CBPALETTE[0]) +
#      geom_point(size=2.5, alpha=0.5, fill=CBPALETTE[0]) + 
#      geom_crossbar(data=(foldchange
#                          .groupby(['spike', 'sample_type'])
#                          .agg({'NT50_post': geometric_mean})
#                          .reset_index()
#                          .dropna()
#                          .replace(dep_virus_replacement)
#                          .replace(sample_type_replacement)
#                         ),
#                    inherit_aes=False,
#                    mapping=aes(x='spike', y='NT50_post', ymin='NT50_post', ymax='NT50_post'),
#                   ) +
#      geom_hline(yintercept=config['NT50_LOD'], 
#                 linetype='dotted', 
#                 size=1, 
#                 alpha=0.6, 
#                 color=CBPALETTE[7]) +
#      theme(figure_size=(0.75*foldchange['spike'].nunique()*foldchange['sample_type'].nunique(),2.5),
#            strip_background=element_blank(),
#            axis_text_x=element_text(angle=90),
#            strip_text_x=element_text(size=12),
#            text=element_text(size=12),
#            ) +
#      facet_wrap('~sample_type')+
#      scale_y_log10(limits=[config['NT50_LOD'],foldchange['NT50_pre'].max()*3.5], 
#                    name='NT50 post-depletion\nof RBD antibodies') +
#      xlab ('')
#      )

# _ = p.draw()
# p.save(f'{resultsdir}/compare_residual_NT50.pdf')
```


```python
# # what if I try plotting the fold-change IC50s?
# p = (ggplot((foldchange
#              .query('depletion=="RBD antibodies depleted"')
#              .drop(columns=['depletion', 'NT50'])
#              .drop_duplicates()
#              .replace(dep_virus_replacement)
#              .replace(sample_type_replacement)
#              .assign(spike=lambda x: pd.Categorical (x['spike'],
#                                                      ordered=True,
#                                                      categories=dep_virus_replacement.values()))
#             )
#            ) +
#      aes('spike', 'fold_change', group='serum', shape='post_ic50_bound') +
#      geom_line(aes(x='spike', y='fold_change', group='serum'), color=CBPALETTE[0]) +
#      geom_point(aes(shape='post_ic50_bound'),size=2.5, alpha=0.5, fill=CBPALETTE[0]) + 
#      geom_crossbar(data=(foldchange
#                          .groupby(['spike', 'sample_type'])
#                          .agg({'fold_change': geometric_mean})
#                          .reset_index()
#                          .dropna()
#                          .replace(dep_virus_replacement)
#                          .replace(sample_type_replacement)
#                         ),
#                    inherit_aes=False,
#                    mapping=aes(x='spike', y='fold_change', ymin='fold_change', ymax='fold_change'),
#                   ) +
#      theme(figure_size=(0.75*foldchange['spike'].nunique()*foldchange['sample_type'].nunique(),2.5),
#            strip_background=element_blank(),
#            axis_text_x=element_text(angle=90),
#            strip_text_x=element_text(size=12),
#            text=element_text(size=12),
#            ) +
#      facet_wrap('~sample_type')+
#      scale_y_log10( #limits=[config['NT50_LOD'],foldchange['NT50_pre'].max()*3.5], 
#                    name='fold-change post-depletion\nof RBD antibodies') +
#      xlab ('')+
#      scale_shape_manual(values=['o','^'],name='NT50 post-depletion\nof RBD antibodies\nNT50 is censored')
#      )

# _ = p.draw()
# p.save(f'{resultsdir}/compare_foldchange.pdf')
```

## Add data from previous studies 
These are:
- early 2020 infections (~30 days post-symptom onset)
- 2x mRNA-1273 vaccinations (~30 days post-dose 1, which is comparable to the 2x BNT162b2 samples we have here)
- Beta primary infections (~30 days post-symptom onset)

I need to wrangle these data frames (which are supplementary files from previous papers) into the same format as what I have here for the Delta study. 


```python
moderna=(pd.read_csv(config['previous_studies_rbd_depletions']['moderna'])
         .assign(virus=lambda x: x['depletion']+' ('+x['spike']+' spike)')
        )

infection=(pd.read_csv(config['previous_studies_rbd_depletions']['infection'])
         .assign(virus=lambda x: x['depletion']+' ('+x['spike']+' spike)')
        )

columns_to_keep=['serum', 'sample_type','spike', 'virus', 'depletion', 
                 'NT50',  'ic50_is_bound','early_late']

combined_df = (pd.concat([foldchange.assign(early_late='day 30-60')[columns_to_keep],
                         moderna[columns_to_keep],
                         infection[columns_to_keep]],
                        ignore_index=True)
               .query('early_late=="day 30-60"') # & spike!="Delta dep x D614G PV"
               .replace(depletion_replacement)
               .assign(depletion=lambda x: pd.Categorical(x['depletion'],
                                                          categories=depletion_replacement.values(),
                                                          ordered=True),
                       sample_type_spike=lambda x: x['sample_type'] + ' ' + x['spike'] + ' spike',
                      )
              )

display(HTML(combined_df.head(2).to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>sample_type</th>
      <th>spike</th>
      <th>virus</th>
      <th>depletion</th>
      <th>NT50</th>
      <th>ic50_is_bound</th>
      <th>early_late</th>
      <th>sample_type_spike</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C</td>
      <td>Delta breakthrough</td>
      <td>D614G</td>
      <td>mock depletion (D614G spike)</td>
      <td>mock</td>
      <td>1550.066341</td>
      <td>False</td>
      <td>day 30-60</td>
      <td>Delta breakthrough D614G spike</td>
    </tr>
    <tr>
      <td>267C</td>
      <td>Delta breakthrough</td>
      <td>D614G</td>
      <td>RBD antibodies depleted (D614G spike)</td>
      <td>depleted</td>
      <td>25.000000</td>
      <td>True</td>
      <td>day 30-60</td>
      <td>Delta breakthrough D614G spike</td>
    </tr>
  </tbody>
</table>



```python
depletions_NT50_LOD = (pd.DataFrame
                       .from_dict(config['depletions_NT50_LOD'], orient='index')
                       .reset_index()
                      )

depletions_NT50_LOD.columns=['sample_type', 'NT50']

depletions_NT50_LOD=(depletions_NT50_LOD
                     .merge(combined_df[['sample_type', 'sample_type_spike']].drop_duplicates())
                    )

display(HTML(depletions_NT50_LOD.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sample_type</th>
      <th>NT50</th>
      <th>sample_type_spike</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Delta breakthrough</td>
      <td>25</td>
      <td>Delta breakthrough D614G spike</td>
    </tr>
    <tr>
      <td>Delta breakthrough</td>
      <td>25</td>
      <td>Delta breakthrough Delta spike</td>
    </tr>
    <tr>
      <td>Delta breakthrough</td>
      <td>25</td>
      <td>Delta breakthrough Delta dep x D614G PV spike</td>
    </tr>
    <tr>
      <td>primary Delta infection</td>
      <td>25</td>
      <td>primary Delta infection D614G spike</td>
    </tr>
    <tr>
      <td>primary Delta infection</td>
      <td>25</td>
      <td>primary Delta infection Delta spike</td>
    </tr>
    <tr>
      <td>primary Delta infection</td>
      <td>25</td>
      <td>primary Delta infection Delta dep x D614G PV spike</td>
    </tr>
    <tr>
      <td>2x BNT162b2</td>
      <td>25</td>
      <td>2x BNT162b2 D614G spike</td>
    </tr>
    <tr>
      <td>2x BNT162b2</td>
      <td>25</td>
      <td>2x BNT162b2 Delta spike</td>
    </tr>
    <tr>
      <td>2x BNT162b2</td>
      <td>25</td>
      <td>2x BNT162b2 Delta dep x D614G PV spike</td>
    </tr>
    <tr>
      <td>2x mRNA-1273</td>
      <td>25</td>
      <td>2x mRNA-1273 D614G spike</td>
    </tr>
    <tr>
      <td>primary Beta infection</td>
      <td>25</td>
      <td>primary Beta infection Beta spike</td>
    </tr>
    <tr>
      <td>early 2020 infection</td>
      <td>20</td>
      <td>early 2020 infection D614G spike</td>
    </tr>
  </tbody>
</table>



```python
combined_df['NT50'].min()
```




    21.31009625




```python
for subset in config['depletion_subsets']:
    
    samples = config['depletion_subsets'][subset].keys()
    rename_samples = config['depletion_subsets'][subset]
    
    df = (combined_df
          .query('sample_type_spike.isin(@samples)')
          .replace(rename_samples)
         )
    
    LOD_df = (depletions_NT50_LOD
              .query('sample_type_spike.isin(@samples)')
              .replace(rename_samples)
             )
    # display(HTML(df.head(1).to_html(index=False)))
    
    NT50_lines = (ggplot(df, aes(x='depletion', y='NT50', group='serum')) + 
                  geom_hline(data=LOD_df,
                             mapping=aes(yintercept='NT50'),
                             color=CBPALETTE[7],
                             alpha=1,
                             size=1,
                             linetype='dotted',
                            ) +
                  geom_point(size=2.5, alpha=0.25) +
                  geom_line(alpha=0.25) +
                  facet_wrap('~sample_type_spike',ncol=5) +
                  theme(axis_title_y=element_text(margin={'r': 6}),
                        strip_background=element_blank(),
                        figure_size=(1.75*df['sample_type_spike'].nunique(),2),
                        axis_text_x=element_text(angle=90),
                        text=element_text(size=12),
                        strip_text_x=element_text(size=12),
                       ) +
                  scale_y_log10(name='neutralization titer (NT50)', limits=[20, combined_df['NT50'].max()]) +
                  xlab('mock or depletion of RBD-binding antibodies')
                 )

    _ = NT50_lines.draw()
    NT50_lines.save(f'{resultsdir}/compare_{subset}.pdf')
```

    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 8.75 x 2 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/compare_vax_vs_infection.pdf
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 3.5 x 2 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/compare_breakthrough.pdf
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 3.5 x 2 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/compare_mix_match.pdf
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 5.25 x 2 in image.
    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: results/rbd_depletion_neuts/compare_heterologous.pdf



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_29_1.png)
    



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_29_2.png)
    



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_29_3.png)
    



    
![png](rbd_depletion_neuts_files/rbd_depletion_neuts_29_4.png)
    



```python

```
