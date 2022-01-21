# Aggregate antibody-escape scores from all studies and libraries
In this repo, we are using antibody-escape scores from multiple projects mapped against multiple different RBD libraries and initially analyzed elsewhere. 

This Python Jupyter notebook aggregates all these antibody-escape scores and makes one supplementary file containing all of them.

## Set up
Import Python modules:


```python
import itertools
import os

from IPython.display import display, HTML

import matplotlib.pyplot as plt

import pandas as pd

from plotnine import *

import yaml
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Create output directory:


```python
os.makedirs(config['supp_data_dir'], exist_ok=True)
```

Extract from configuration what we will use as the site- and mutation-level metrics:


```python
escape_frac_types = config['escape_frac_files']
escape_frac_files = [config[f] for f in escape_frac_types]
site_metrics = {config[f]:config[f'{f[:-12]}site_metric'] for f in escape_frac_types}
mut_metrics = {config[f]:config[f'{f[:-12]}mut_metric'] for f in escape_frac_types}
escape_frac_libraries = config['escape_frac_libraries']

print(escape_frac_types)
print('At site level, quantifying selection by:')
for f in escape_frac_types:
    print(f'\t{f}: {site_metrics[config[f]]}')

print('At mutation level, quantifying selection by:')
for f in escape_frac_types:
    print(f'\t{f}: {mut_metrics[config[f]]}')
```

    ['escape_fracs', 'early2020_escape_fracs', 'beta_escape_fracs']
    At site level, quantifying selection by:
    	escape_fracs: site_total_escape_frac_single_mut
    	early2020_escape_fracs: site_total_escape_frac_epistasis_model
    	beta_escape_fracs: site_total_escape_frac_single_mut
    At mutation level, quantifying selection by:
    	escape_fracs: mut_escape_frac_single_mut
    	early2020_escape_fracs: mut_escape_frac_epistasis_model
    	beta_escape_fracs: mut_escape_frac_single_mut


Read the escape fractions


```python
escape_fracs_dfs = []

for f,l in zip(escape_frac_files, escape_frac_types):
    
    site_metric = site_metrics[f]
    mut_metric = mut_metrics[f]
    
    library = escape_frac_libraries[l]

    df = (pd.read_csv(f)
          .query('library == "average"')
          .drop(columns=['site', 'selection', 'library'])
          .rename(columns={'label_site': 'site',
                           site_metric: 'site_total_escape',
                           mut_metric: 'mut_escape'
                          }
                 )
          [['condition', 'site', 'wildtype', 'mutation', 'mut_escape', 'site_total_escape']]
          .assign(site_max_escape=lambda x: x.groupby(['condition', 'site'])['mut_escape'].transform('max'),
                  library=library,
                 )
         )
    escape_fracs_dfs.append(df)

escape_fracs=(pd.concat(escape_fracs_dfs)
              .merge(pd.read_csv(config['aggregate_escape_scores_metadata_file']),
                     how='right',
                     on='condition',
                     validate='many_to_one'
                    )
              .assign(condition=lambda x: x['name'])
              .drop(columns=['name'])
             )

# pd.read_csv(config['aggregate_escape_scores_metadata_file'])

print('First few lines of escape-fraction data frame:')
display(HTML(escape_fracs.head().to_html(index=False)))

csv_file = config['aggregate_escape_scores_file']
print(f"Writing to {csv_file}")
escape_fracs.to_csv(csv_file, index=False, float_format='%.4g')
```

    First few lines of escape-fraction data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>site</th>
      <th>wildtype</th>
      <th>mutation</th>
      <th>mut_escape</th>
      <th>site_total_escape</th>
      <th>site_max_escape</th>
      <th>library</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>P02</td>
      <td>331</td>
      <td>N</td>
      <td>A</td>
      <td>0.001502</td>
      <td>0.0456</td>
      <td>0.006194</td>
      <td>Delta</td>
      <td>2x BNT162b2</td>
    </tr>
    <tr>
      <td>P02</td>
      <td>331</td>
      <td>N</td>
      <td>C</td>
      <td>0.001763</td>
      <td>0.0456</td>
      <td>0.006194</td>
      <td>Delta</td>
      <td>2x BNT162b2</td>
    </tr>
    <tr>
      <td>P02</td>
      <td>331</td>
      <td>N</td>
      <td>D</td>
      <td>0.001599</td>
      <td>0.0456</td>
      <td>0.006194</td>
      <td>Delta</td>
      <td>2x BNT162b2</td>
    </tr>
    <tr>
      <td>P02</td>
      <td>331</td>
      <td>N</td>
      <td>E</td>
      <td>0.002393</td>
      <td>0.0456</td>
      <td>0.006194</td>
      <td>Delta</td>
      <td>2x BNT162b2</td>
    </tr>
    <tr>
      <td>P02</td>
      <td>331</td>
      <td>N</td>
      <td>F</td>
      <td>0.001360</td>
      <td>0.0456</td>
      <td>0.006194</td>
      <td>Delta</td>
      <td>2x BNT162b2</td>
    </tr>
  </tbody>
</table>


    Writing to results/supp_data/aggregate_raw_data.csv



```python

```
