# Output PDBs with escape scores as B factors
This Python Jupyter notebook outputs PDBs with the escape scores as B factors.

Though we will want more elaborate series of commands to codify our visualization of these RBD structures colored by escape, the series of commands below, when executed in a `PyMol` session with one of these PDBs open, will color the RBD surface according to escape scores.

For example, to normalize each structure colored by the max mut effect, we might want to have a white to red scale from 0 to 1:

     create RBD, chain E
     hide all; show cartoon, chain A; color gray20, chain A
     show surface, RBD; spectrum b, white red, RBD, minimum=0, maximum=1
     
For something like total escape, maybe we want each structure normalized to the maximum total escape in that structure, in which case we can just leave the maximum argument empty:

     create RBD, chain E
     hide all; show cartoon, chain A; color gray20, chain A
     show surface, RBD; spectrum b, white red, RBD, minimum=0
     
We write PDBs with B factors indicating the total site escape and maximum mutation escape at each site, and the same with these values normalized to the maximum for the full structure (the latter are easier to process in `Chimera`).

First, import Python modules:


```python
import collections
import copy
import os
import warnings

import Bio.PDB

import dms_variants.pdb_utils

from IPython.display import display, HTML

import pandas as pd

import yaml
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Read configuration for outputting PDBs:


```python
print(f"Reading PDB output configuration from {config['output_pdbs_config']}")
with open(config['output_pdbs_config']) as f:
    output_pdbs_config = yaml.safe_load(f)
```

    Reading PDB output configuration from data/output_pdbs_config.yaml


Make output directory:


```python
os.makedirs(config['pdb_outputs_dir'], exist_ok=True)
```

Read escape fractions and compute **total** and **maximum** escape at each site, and also the total and maximum escape at each site normalized to be between 0 and 1 for each selection:


```python
print(f"Reading escape fractions from {config['escape_fracs']}")

escape_fracs = (
    pd.read_csv(config['escape_fracs'])
    .query('library == "average"')
    .assign(site=lambda x: x['label_site'])
    .groupby(['selection', 'site'])
    .aggregate(total_escape=pd.NamedAgg(config['mut_metric'], 'sum'),
               max_escape=pd.NamedAgg(config['mut_metric'], 'max')
               )
    .reset_index()
    .assign(max_total_escape=lambda x: x.groupby('selection')['total_escape'].transform('max'),
            max_max_escape=lambda x: x.groupby('selection')['max_escape'].transform('max'),
            norm_total_escape=lambda x: x['total_escape'] / x['max_total_escape'],
            norm_max_escape=lambda x: x['max_escape'] / x['max_max_escape'])
    )

display(HTML(escape_fracs.head().to_html(index=False)))
```

    Reading escape fractions from results/escape_scores/escape_fracs.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>selection</th>
      <th>site</th>
      <th>total_escape</th>
      <th>max_escape</th>
      <th>max_total_escape</th>
      <th>max_max_escape</th>
      <th>norm_total_escape</th>
      <th>norm_max_escape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_repeat_200</td>
      <td>331</td>
      <td>0.053354</td>
      <td>0.005160</td>
      <td>2.49289</td>
      <td>0.4069</td>
      <td>0.021402</td>
      <td>0.012681</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>332</td>
      <td>0.077189</td>
      <td>0.007509</td>
      <td>2.49289</td>
      <td>0.4069</td>
      <td>0.030964</td>
      <td>0.018454</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>333</td>
      <td>0.056405</td>
      <td>0.005939</td>
      <td>2.49289</td>
      <td>0.4069</td>
      <td>0.022626</td>
      <td>0.014596</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>334</td>
      <td>0.125584</td>
      <td>0.020500</td>
      <td>2.49289</td>
      <td>0.4069</td>
      <td>0.050377</td>
      <td>0.050381</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>335</td>
      <td>0.101017</td>
      <td>0.014700</td>
      <td>2.49289</td>
      <td>0.4069</td>
      <td>0.040522</td>
      <td>0.036127</td>
    </tr>
  </tbody>
</table>


Now map the escape metrics to the B-factors.
For sites where no mutations have escape scores:
 - In the RBD chain(s) fill the B-factor for non-normalized scores to -1 to enable collapsing to zero or callout as a a separate class, depending how we choose to color sites for different visualizations. For normalized scores, fill to 0.
 - In other chains, always fill missing B factors to 0.  


```python
for name, specs in output_pdbs_config.items():
    print(f"\nMaking PDB mappings for {name} to {specs['pdbfile']}")
    assert os.path.isfile(specs['pdbfile'])
    
    # get escape fracs just for conditions of interest
    if isinstance(specs['conditions'], str) and specs['conditions'].upper() == 'ALL':
        conditions = escape_fracs['selection'].unique().tolist()
    else:
        assert isinstance(specs['conditions'], list)
        conditions = specs['conditions']
    print(f"Making mappings for {len(conditions)} conditions.")
    df = escape_fracs.query('selection in @conditions')
    
    # get chains
    assert isinstance(specs['chains'], list)
    print('Mapping to the following chains: ' + ', '.join(specs['chains']))
    df = pd.concat([df.assign(chain=chain) for chain in specs['chains']], ignore_index=True)
    
    # make mappings for each condition and metric
    for condition, df in df.groupby('selection'):
        print(f"  Writing B-factor re-assigned PDBs for {condition} to:")
    
        for metric in ['total_escape', 'max_escape', 'norm_total_escape', 'norm_max_escape']:
        
            # what do we assign to missing sites?
            missing_metric = collections.defaultdict(lambda: 0)  # non-RBD chains always fill to zero
            for chain in specs['chains']:
                if 'norm' in metric:
                    missing_metric[chain] = 0  # missing sites in RBD are 0 for normalized metric PDBs
                else:
                    missing_metric[chain] = -1  # missing sites in RBD are -1 for non-normalized metric PDBs
        
            fname = os.path.join(config['pdb_outputs_dir'], f"{condition}_{name}_{metric}.pdb")
            print(f"    {fname}")
            
            dms_variants.pdb_utils.reassign_b_factor(input_pdbfile=specs['pdbfile'],
                                                     output_pdbfile=fname,
                                                     df=df,
                                                     metric_col=metric,
                                                     missing_metric=missing_metric)
```

    
    Making PDB mappings for 6m0j to data/pdbs/6M0J.pdb
    Making mappings for 24 conditions.
    Mapping to the following chains: E
      Writing B-factor re-assigned PDBs for 267C_repeat_200 to:
        results/pdb_outputs/267C_repeat_200_6m0j_total_escape.pdb
        results/pdb_outputs/267C_repeat_200_6m0j_max_escape.pdb
        results/pdb_outputs/267C_repeat_200_6m0j_norm_total_escape.pdb
        results/pdb_outputs/267C_repeat_200_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 268C_repeat_500 to:
        results/pdb_outputs/268C_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/268C_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/268C_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/268C_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 273C_repeat_500 to:
        results/pdb_outputs/273C_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/273C_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/273C_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/273C_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 274C_repeat_500 to:
        results/pdb_outputs/274C_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/274C_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/274C_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/274C_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 276C_repeat_500 to:
        results/pdb_outputs/276C_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/276C_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/276C_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/276C_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 277C_repeat_500 to:
        results/pdb_outputs/277C_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/277C_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/277C_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/277C_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 278C_repeat_1250 to:
        results/pdb_outputs/278C_repeat_1250_6m0j_total_escape.pdb
        results/pdb_outputs/278C_repeat_1250_6m0j_max_escape.pdb
        results/pdb_outputs/278C_repeat_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/278C_repeat_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 279C_repeat_1250 to:
        results/pdb_outputs/279C_repeat_1250_6m0j_total_escape.pdb
        results/pdb_outputs/279C_repeat_1250_6m0j_max_escape.pdb
        results/pdb_outputs/279C_repeat_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/279C_repeat_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_10_1250 to:
        results/pdb_outputs/Delta_10_1250_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_10_1250_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_10_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_10_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_11_500 to:
        results/pdb_outputs/Delta_11_500_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_11_500_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_11_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_11_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_1_500 to:
        results/pdb_outputs/Delta_1_500_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_1_500_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_1_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_1_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_3_350 to:
        results/pdb_outputs/Delta_3_350_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_3_350_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_3_350_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_3_350_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_4_350 to:
        results/pdb_outputs/Delta_4_350_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_4_350_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_4_350_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_4_350_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_6_500 to:
        results/pdb_outputs/Delta_6_500_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_6_500_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_6_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_6_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_7_1250 to:
        results/pdb_outputs/Delta_7_1250_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_7_1250_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_7_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_7_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_8_500 to:
        results/pdb_outputs/Delta_8_500_6m0j_total_escape.pdb
        results/pdb_outputs/Delta_8_500_6m0j_max_escape.pdb
        results/pdb_outputs/Delta_8_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/Delta_8_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P02_repeat_500 to:
        results/pdb_outputs/P02_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/P02_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/P02_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P02_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P03_repeat_1250 to:
        results/pdb_outputs/P03_repeat_1250_6m0j_total_escape.pdb
        results/pdb_outputs/P03_repeat_1250_6m0j_max_escape.pdb
        results/pdb_outputs/P03_repeat_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P03_repeat_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P04_repeat_1250 to:
        results/pdb_outputs/P04_repeat_1250_6m0j_total_escape.pdb
        results/pdb_outputs/P04_repeat_1250_6m0j_max_escape.pdb
        results/pdb_outputs/P04_repeat_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P04_repeat_1250_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P05_repeat_500 to:
        results/pdb_outputs/P05_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/P05_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/P05_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P05_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P08_repeat_500 to:
        results/pdb_outputs/P08_repeat_500_6m0j_total_escape.pdb
        results/pdb_outputs/P08_repeat_500_6m0j_max_escape.pdb
        results/pdb_outputs/P08_repeat_500_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P08_repeat_500_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P09_repeat_200 to:
        results/pdb_outputs/P09_repeat_200_6m0j_total_escape.pdb
        results/pdb_outputs/P09_repeat_200_6m0j_max_escape.pdb
        results/pdb_outputs/P09_repeat_200_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P09_repeat_200_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P12_repeat_200 to:
        results/pdb_outputs/P12_repeat_200_6m0j_total_escape.pdb
        results/pdb_outputs/P12_repeat_200_6m0j_max_escape.pdb
        results/pdb_outputs/P12_repeat_200_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P12_repeat_200_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P14_repeat_1250 to:
        results/pdb_outputs/P14_repeat_1250_6m0j_total_escape.pdb
        results/pdb_outputs/P14_repeat_1250_6m0j_max_escape.pdb
        results/pdb_outputs/P14_repeat_1250_6m0j_norm_total_escape.pdb
        results/pdb_outputs/P14_repeat_1250_6m0j_norm_max_escape.pdb
    
    Making PDB mappings for 7V8B to data/pdbs/7V8B.pdb
    Making mappings for 24 conditions.
    Mapping to the following chains: A
      Writing B-factor re-assigned PDBs for 267C_repeat_200 to:
        results/pdb_outputs/267C_repeat_200_7V8B_total_escape.pdb
        results/pdb_outputs/267C_repeat_200_7V8B_max_escape.pdb
        results/pdb_outputs/267C_repeat_200_7V8B_norm_total_escape.pdb
        results/pdb_outputs/267C_repeat_200_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 268C_repeat_500 to:
        results/pdb_outputs/268C_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/268C_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/268C_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/268C_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 273C_repeat_500 to:
        results/pdb_outputs/273C_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/273C_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/273C_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/273C_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 274C_repeat_500 to:
        results/pdb_outputs/274C_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/274C_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/274C_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/274C_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 276C_repeat_500 to:
        results/pdb_outputs/276C_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/276C_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/276C_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/276C_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 277C_repeat_500 to:
        results/pdb_outputs/277C_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/277C_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/277C_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/277C_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 278C_repeat_1250 to:
        results/pdb_outputs/278C_repeat_1250_7V8B_total_escape.pdb
        results/pdb_outputs/278C_repeat_1250_7V8B_max_escape.pdb
        results/pdb_outputs/278C_repeat_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/278C_repeat_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for 279C_repeat_1250 to:
        results/pdb_outputs/279C_repeat_1250_7V8B_total_escape.pdb
        results/pdb_outputs/279C_repeat_1250_7V8B_max_escape.pdb
        results/pdb_outputs/279C_repeat_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/279C_repeat_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_10_1250 to:
        results/pdb_outputs/Delta_10_1250_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_10_1250_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_10_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_10_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_11_500 to:
        results/pdb_outputs/Delta_11_500_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_11_500_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_11_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_11_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_1_500 to:
        results/pdb_outputs/Delta_1_500_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_1_500_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_1_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_1_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_3_350 to:
        results/pdb_outputs/Delta_3_350_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_3_350_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_3_350_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_3_350_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_4_350 to:
        results/pdb_outputs/Delta_4_350_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_4_350_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_4_350_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_4_350_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_6_500 to:
        results/pdb_outputs/Delta_6_500_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_6_500_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_6_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_6_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_7_1250 to:
        results/pdb_outputs/Delta_7_1250_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_7_1250_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_7_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_7_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for Delta_8_500 to:
        results/pdb_outputs/Delta_8_500_7V8B_total_escape.pdb
        results/pdb_outputs/Delta_8_500_7V8B_max_escape.pdb
        results/pdb_outputs/Delta_8_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/Delta_8_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P02_repeat_500 to:
        results/pdb_outputs/P02_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/P02_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/P02_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P02_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P03_repeat_1250 to:
        results/pdb_outputs/P03_repeat_1250_7V8B_total_escape.pdb
        results/pdb_outputs/P03_repeat_1250_7V8B_max_escape.pdb
        results/pdb_outputs/P03_repeat_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P03_repeat_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P04_repeat_1250 to:
        results/pdb_outputs/P04_repeat_1250_7V8B_total_escape.pdb
        results/pdb_outputs/P04_repeat_1250_7V8B_max_escape.pdb
        results/pdb_outputs/P04_repeat_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P04_repeat_1250_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P05_repeat_500 to:
        results/pdb_outputs/P05_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/P05_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/P05_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P05_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P08_repeat_500 to:
        results/pdb_outputs/P08_repeat_500_7V8B_total_escape.pdb
        results/pdb_outputs/P08_repeat_500_7V8B_max_escape.pdb
        results/pdb_outputs/P08_repeat_500_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P08_repeat_500_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P09_repeat_200 to:
        results/pdb_outputs/P09_repeat_200_7V8B_total_escape.pdb
        results/pdb_outputs/P09_repeat_200_7V8B_max_escape.pdb
        results/pdb_outputs/P09_repeat_200_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P09_repeat_200_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P12_repeat_200 to:
        results/pdb_outputs/P12_repeat_200_7V8B_total_escape.pdb
        results/pdb_outputs/P12_repeat_200_7V8B_max_escape.pdb
        results/pdb_outputs/P12_repeat_200_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P12_repeat_200_7V8B_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for P14_repeat_1250 to:
        results/pdb_outputs/P14_repeat_1250_7V8B_total_escape.pdb
        results/pdb_outputs/P14_repeat_1250_7V8B_max_escape.pdb
        results/pdb_outputs/P14_repeat_1250_7V8B_norm_total_escape.pdb
        results/pdb_outputs/P14_repeat_1250_7V8B_norm_max_escape.pdb

