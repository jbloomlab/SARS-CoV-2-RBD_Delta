# Calculate titers of RBD mutant spike-pseudotyped lentiviruses
Experiments by Rachel Eguia.


```python
import os
import warnings

import math
import numpy as np 

from IPython.display import display, HTML
import matplotlib.pyplot as plt

from neutcurve.colorschemes import CBMARKERS, CBPALETTE

import pandas as pd
from plotnine import *
```


```python
warnings.simplefilter('ignore')
```

Make output directory if needed


```python
resultsdir = './results/entry_titers'
os.makedirs(resultsdir, exist_ok=True)
```


```python
titerdir = 'data/entry_titers/'
titers = pd.DataFrame() # create empty data frame

for f in os.listdir(titerdir):
    titerfile = os.path.join(titerdir, f)
    print(titerfile)
    titers = titers.append(pd.read_csv(titerfile)).reset_index(drop=True)
    
titers = (titers
          .assign(RLUperuL=lambda x: x['RLU_per_well'] / x['uL_virus'],
                  date=lambda x: x['date'].astype(str),
                  date_rescued=lambda x: x['date_rescued'].astype(str)
                 )
         )

display(HTML(titers.head().to_html(index=False)))
```

    data/entry_titers/12Nov21_infection_titers.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>plasmid</th>
      <th>replicate</th>
      <th>virus</th>
      <th>dilution</th>
      <th>uL_virus</th>
      <th>RLU_per_well</th>
      <th>date</th>
      <th>date_rescued</th>
      <th>cell_type</th>
      <th>RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NaN</td>
      <td>rep1</td>
      <td>Delta_WT</td>
      <td>0.10000</td>
      <td>10.000</td>
      <td>344077</td>
      <td>12-Nov-21</td>
      <td>5-Nov-21</td>
      <td>293T-ACE2-TMPRSS2</td>
      <td>34407.7</td>
    </tr>
    <tr>
      <td>NaN</td>
      <td>rep1</td>
      <td>Delta_WT</td>
      <td>0.05000</td>
      <td>5.000</td>
      <td>207846</td>
      <td>12-Nov-21</td>
      <td>5-Nov-21</td>
      <td>293T-ACE2-TMPRSS2</td>
      <td>41569.2</td>
    </tr>
    <tr>
      <td>NaN</td>
      <td>rep1</td>
      <td>Delta_WT</td>
      <td>0.02500</td>
      <td>2.500</td>
      <td>59660</td>
      <td>12-Nov-21</td>
      <td>5-Nov-21</td>
      <td>293T-ACE2-TMPRSS2</td>
      <td>23864.0</td>
    </tr>
    <tr>
      <td>NaN</td>
      <td>rep1</td>
      <td>Delta_WT</td>
      <td>0.01250</td>
      <td>1.250</td>
      <td>33656</td>
      <td>12-Nov-21</td>
      <td>5-Nov-21</td>
      <td>293T-ACE2-TMPRSS2</td>
      <td>26924.8</td>
    </tr>
    <tr>
      <td>NaN</td>
      <td>rep1</td>
      <td>Delta_WT</td>
      <td>0.00625</td>
      <td>0.625</td>
      <td>13816</td>
      <td>12-Nov-21</td>
      <td>5-Nov-21</td>
      <td>293T-ACE2-TMPRSS2</td>
      <td>22105.6</td>
    </tr>
  </tbody>
</table>



```python
print(titers['virus'].unique())
```

    ['Delta_WT' 'Delta_E484K' 'Delta_K444N' 'Delta_G446V' 'Delta_S494L'
     'Delta_K417N' 'Delta_E484Q' 'VSV-G' 'No_VEP' 'Delta_R452L' 'Delta_R346G']



```python
ncol=min(8, titers['virus'].nunique())
nrow=math.ceil(titers['virus'].nunique() / ncol)

p = (ggplot(titers.drop(columns=['plasmid']).dropna()
            ) +
     aes('uL_virus', 'RLU_per_well', group='replicate') +
     geom_point(size=1.5) +
     geom_line() +
     facet_wrap('~virus+date_rescued', ncol=ncol) +
     scale_y_log10(name='RLU per well') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(2 * ncol, 1.75 * nrow),
           )
     )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'RLU-vs-uL.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to ./results/entry_titers/RLU-vs-uL.pdf



    
![png](calculate_entry_titer_files/calculate_entry_titer_7_1.png)
    



```python
p = (ggplot(titers.drop(columns=['plasmid']).dropna()
            ) +
     aes('uL_virus', 'RLUperuL', group='replicate') +
     geom_point(size=1.5) +
     geom_line() +
     facet_wrap('~virus+date_rescued', ncol=ncol) +
     scale_y_log10(name='RLU per uL') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(2 * ncol, 1.75 * nrow),
           ) 
     )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'RLUperuL.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to ./results/entry_titers/RLUperuL.pdf



    
![png](calculate_entry_titer_files/calculate_entry_titer_8_1.png)
    


From visual inspection of the above plots, it appears that only the 5 highest dilutions (i.e., >0.5uL of virus per well) are reliable enough to calculate titers. 


```python
average_titers = (titers
                  .query('uL_virus > 0.5') # drop lowest concentration of virus
                  .groupby(['virus', 'replicate', 'date_rescued'])
                  .agg(mean_RLUperuL=pd.NamedAgg(column='RLUperuL', aggfunc=np.mean))
                  .reset_index()
                 )

display(HTML(average_titers.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>virus</th>
      <th>replicate</th>
      <th>date_rescued</th>
      <th>mean_RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Delta_E484K</td>
      <td>rep1</td>
      <td>25-Oct-21</td>
      <td>18584.98</td>
    </tr>
    <tr>
      <td>Delta_E484K</td>
      <td>rep1</td>
      <td>5-Nov-21</td>
      <td>36906.40</td>
    </tr>
    <tr>
      <td>Delta_E484K</td>
      <td>rep2</td>
      <td>25-Oct-21</td>
      <td>23217.76</td>
    </tr>
    <tr>
      <td>Delta_E484K</td>
      <td>rep2</td>
      <td>5-Nov-21</td>
      <td>30786.36</td>
    </tr>
    <tr>
      <td>Delta_E484Q</td>
      <td>rep1</td>
      <td>25-Oct-21</td>
      <td>4195.08</td>
    </tr>
  </tbody>
</table>



```python
p = (ggplot(average_titers, 
            aes(x='virus', y='mean_RLUperuL', color='date_rescued')
           ) +
     geom_point(size=2.5, alpha=0.5)+
     theme_classic() +
     theme(axis_text_x=element_text(angle=90, vjust=1, hjust=0.5),
           figure_size=(average_titers['virus'].nunique()*0.35,2),
           axis_title_x=element_blank(),
          ) +
     scale_y_log10(limits=[1,1.1e6]) +
     ylab('relative luciferase units\nper uL')+
     labs(title='pseudovirus entry titers') +
     scale_color_manual(values=CBPALETTE)
    )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'entry_titers.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to ./results/entry_titers/entry_titers.pdf



    
![png](calculate_entry_titer_files/calculate_entry_titer_11_1.png)
    


Calculate how much virus to use in neut assays:


```python
target_RLU = 2.5e5
uL_virus_per_well = 50

dilute_virus = (average_titers
                .groupby(['virus', 'date_rescued'])
                .agg(RLUperuL=pd.NamedAgg(column='mean_RLUperuL', aggfunc=np.mean))
                .reset_index()
                .assign(target_RLU = target_RLU,
                        uL_virus_per_well = uL_virus_per_well,
                        dilution_factor = lambda x: x['RLUperuL']/target_RLU*uL_virus_per_well,
                        uL_per_6mL = lambda x: 6000/x['dilution_factor']
                       )
               )


titerfile = os.path.join(resultsdir, 'titers.csv')
print(f"Saving to {titerfile}")

dilute_virus.to_csv(titerfile, index=False)

display(HTML(dilute_virus.head().to_html(index=False)))
```

    Saving to ./results/entry_titers/titers.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>virus</th>
      <th>date_rescued</th>
      <th>RLUperuL</th>
      <th>target_RLU</th>
      <th>uL_virus_per_well</th>
      <th>dilution_factor</th>
      <th>uL_per_6mL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Delta_E484K</td>
      <td>25-Oct-21</td>
      <td>20901.37</td>
      <td>250000.0</td>
      <td>50</td>
      <td>4.180274</td>
      <td>1435.312613</td>
    </tr>
    <tr>
      <td>Delta_E484K</td>
      <td>5-Nov-21</td>
      <td>33846.38</td>
      <td>250000.0</td>
      <td>50</td>
      <td>6.769276</td>
      <td>886.357714</td>
    </tr>
    <tr>
      <td>Delta_E484Q</td>
      <td>25-Oct-21</td>
      <td>4640.78</td>
      <td>250000.0</td>
      <td>50</td>
      <td>0.928156</td>
      <td>6464.430548</td>
    </tr>
    <tr>
      <td>Delta_E484Q</td>
      <td>5-Nov-21</td>
      <td>24338.88</td>
      <td>250000.0</td>
      <td>50</td>
      <td>4.867776</td>
      <td>1232.595748</td>
    </tr>
    <tr>
      <td>Delta_G446V</td>
      <td>25-Oct-21</td>
      <td>29982.01</td>
      <td>250000.0</td>
      <td>50</td>
      <td>5.996402</td>
      <td>1000.600026</td>
    </tr>
  </tbody>
</table>



```python

```
