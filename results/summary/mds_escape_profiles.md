# Multidimensional scaling of antibody escape profiles
This Python Jupyter notebook performs multi-dimensional scaling of escape profiles to project the antibodies into two dimensions based on similarity of their escape profiles.

## Set up analysis
Import Python modules:


```python
import itertools
import os

import adjustText

from dms_variants.constants import CBPALETTE

from IPython.display import display, HTML

import matplotlib
import matplotlib.pyplot as plt

import numpy

import pandas as pd

import seaborn

import sklearn.manifold
from sklearn.metrics import euclidean_distances

import yaml
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Create output directory:


```python
os.makedirs(config['mds_dir'], exist_ok=True)
```

Extract from configuration what we will use as the site- and mutation-level metrics:


```python
escape_frac_types = config['escape_frac_files']
escape_frac_files = [config[f] for f in escape_frac_types]
site_metrics = {config[f]:config[f'{f[:-12]}site_metric'] for f in escape_frac_types}

print(f"At site level, quantifying selection by {site_metrics}")
```

    At site level, quantifying selection by {'results/escape_scores/escape_fracs.csv': 'site_total_escape_frac_single_mut', 'results/prior_DMS_data/early2020_escape_fracs.csv': 'site_total_escape_frac_epistasis_model', 'results/prior_DMS_data/beta_escape_fracs.csv': 'site_total_escape_frac_single_mut'}


## Read samples and escape fractions
Read the escape fractions.
We only retain the **average** of the libraries for plotting here, not the individual libraries.
Also, we work in the full-Spike rather than RBD numbering, which means we use `label_site` as `site` (and so rename as such below):


```python
escape_fracs_dfs = []

for f in escape_frac_files:
    
    site_metric = site_metrics[f]

    df = (pd.read_csv(f)
          .query('library == "average"')
          .drop(columns=['site', 'selection', 'library'])
          .rename(columns={'label_site': 'site',
                           site_metric: 'site_escape'
                          }
                 )
          [['condition', 'site', 'site_escape']]
         )
    escape_fracs_dfs.append(df)

escape_fracs=pd.concat(escape_fracs_dfs)

# because we are using different site metrics depending on the original study
site_metric='site_escape'
```

## Get antibody sets for each multidimensional scaling
We have manually specified configurations for the MDS plots in a YAML file.
We will do multi-dimensional scaling for each antibody/sera set specified in this file:


```python
print(f"Reading MDS configuration from {config['mds_config']}")
with open(config['mds_config']) as f:
    mds_config = yaml.safe_load(f)
    
print(f"Reading the site color schemes from {config['site_color_schemes']}")
site_color_schemes = pd.read_csv(config['site_color_schemes'])
```

    Reading MDS configuration from data/mds_config.yaml
    Reading the site color schemes from data/site_color_schemes.csv


## Multidimensional scaling
Note that there are three main steps here:
 1. Calculate similarities between profiles of each antibody.
 2. Convert similarities to dissimilarities.
 3. Do multi-dimensional scaling and plot the results.

First, define a function to compute the similarity between all pairs of escape profiles in a data frame.
We calculate similarity as the dot product of the escape profiles for each pair of conditions, using the site-level metric and normalizing each profile so it's dot product with itself is one.
Importantly, we raise the site-level metric to the $p$ power in order to emphasize sites with large values (essentially a p-norm):


```python
def escape_similarity(df, p=1):
    """Compute similarity between all pairs of conditions in `df`."""
    df = df[['condition', 'site', site_metric]].drop_duplicates()
    assert not df.isnull().any().any()
    
    conditions = df['condition'].unique()
    similarities = []
    pivoted_df = (
        df
        .assign(metric=lambda x: x[site_metric]**p)
        .pivot_table(index='site', columns='condition', values='metric', fill_value=0)
        # for normalization: https://stackoverflow.com/a/58113206
        # to get norm: https://stackoverflow.com/a/47953601
        .transform(lambda x: x / numpy.linalg.norm(x, axis=0))
        )
    for cond1, cond2 in itertools.product(conditions, conditions):
        similarity = (
            pivoted_df
            .assign(similarity=lambda x: x[cond1] * x[cond2])
            ['similarity']
            )
        assert similarity.notnull().all()  # make sure no sites have null values
        similarities.append(similarity.sum())  # sum of similarities over sites
    return pd.DataFrame(numpy.array(similarities).reshape(len(conditions), len(conditions)),
                        columns=conditions, index=conditions)
```

Define function to compute dissimilarity $d$ from the similarity $s$.
Options are:
  - **one_minus**: $d = 1 - s$
  - **minus_log**: $d = -\ln s$


```python
def dissimilarity(similarity, method='one_minus'):
    if method == 'one_minus':
        return 1 - similarity
    elif method == 'minus_log':
        return -numpy.log(similarity)
    else:
        raise ValueError(f"invalid `method` {method}")
```

Now compute the similarities and dissimilarities, and do the multidimensional scaling [as described here](https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py).
We do this just for the antibody combinations for which such a plot is specified in the escape profiles configuration file.
We then plot the multidimensional scaling, using [adjustTexts](https://adjusttext.readthedocs.io/) to repel the labels and following [here](https://stackoverflow.com/q/56337732) to draw pie charts that color the points according to the site-coloring scheme if specified in configuration.
These pie charts color by the fraction of the squared site escape apportioned to each site category.


```python
# which method do we use to compute dissimilarity?
dissimilarity_method = 'one_minus'

# do we also plot similarity / dissimilarity matrices?
plot_similarity = False

# function to draw colored pie for each point.
def draw_pie(dist, xpos, ypos, size, ax, colors, alpha, circle_color):
    """Based on this: https://stackoverflow.com/q/56337732"""
    # for incremental pie slices
    cumsum = numpy.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()

    assert len(colors) == len(dist)
    for r1, r2, color in zip(pie[:-1], pie[1:], colors):
        angles = numpy.linspace(2 * numpy.pi * r1, 2 * numpy.pi * r2)
        x = [0] + numpy.cos(angles).tolist()
        y = [0] + numpy.sin(angles).tolist()

        xy = numpy.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, s=size, facecolors=color, alpha=alpha, edgecolors='none')
        ax.scatter(xpos, ypos, marker='o', s=size, edgecolors=circle_color,
                   facecolors='none', alpha=alpha)

    return ax

# loop over combinations to plot
for name, specs in mds_config.items():
    
    # get data frame with just the conditions we want to plot, also re-naming them
    conditions_to_plot = list(specs['conditions'].keys())
    print(f"\nMaking plot {name}, which has the following antibodies:\n{conditions_to_plot}")
    assert len(conditions_to_plot) == len(set(specs['conditions'].values()))
    assert set(conditions_to_plot).issubset(set(escape_fracs['condition']))
    df = (escape_fracs
          .query('condition in @conditions_to_plot')
          .assign(condition=lambda x: x['condition'].map(specs['conditions']))
          )
    
    # compute similarities and dissimilarities
    similarities = escape_similarity(df)
    dissimilarities = similarities.applymap(lambda x: dissimilarity(x, method=dissimilarity_method))
    conditions = df['condition'].unique()
    assert all(conditions == similarities.columns) and all(conditions == similarities.index)
    n = len(conditions)
    
    # plot similarities
    if plot_similarity:
        for title, data in [('Similarities', similarities), ('Dissimilarities', dissimilarities)]:
            fig, ax = plt.subplots(figsize=(0.8 * n, 0.7 * n))
            _ = seaborn.heatmap(data, annot=True, ax=ax)
            plt.title(f"{title} for {name}", size=16)
            plt.show(fig)
            plt.close(fig)
    
    # use multidimensional scaling to get locations of antibodies
    mds = sklearn.manifold.MDS(n_components=2,
                               metric=True,
                               max_iter=3000,
                               eps=1e-6,
                               random_state=1 if 'random_state' not in specs else specs['random_state'],
                               dissimilarity='precomputed',
                               n_jobs=1)
    locs = mds.fit_transform(dissimilarities)
    
    print(f"stress = {getattr(mds,'stress_')} from iteration {getattr(mds,'n_iter_')}")
    
    # the `sklearn` stress is not scaled, so we manually calculate a scaled Kruskal stress
    # as shown here: https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python/64271501#64271501
    # manually calculate stress (unscaled)
    points = mds.embedding_
    DE = euclidean_distances(points)
    stress = 0.5 * numpy.sum((DE - dissimilarities.values)**2)
    print(f"Manual calculus of sklearn stress : {stress}")
    
    ## Kruskal's stress (or stress formula 1)
    stress1 = numpy.sqrt(stress / (0.5 * numpy.sum(dissimilarities.values**2)))
    print(f"Kruskal's Stress : {stress1}")
    print("[Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]")

    # get the colors for each point if relevant
    color_scheme = specs['color_scheme']
    if isinstance(color_scheme, list):
        color_csv, color_col = color_scheme
        print(f"Using condition-level color scheme in column {color_col} of {color_csv}")
        dists = [[1] for condition in conditions]
        condition_to_color = pd.read_csv(color_csv).set_index('condition')[color_col].to_dict()
        if not set(conditions).issubset(set(condition_to_color)):
            raise ValueError(f"{color_scheme} doesn't have colors for all conditions: {conditions}")
        colors = [[condition_to_color[condition]] for condition in conditions]
    elif color_scheme in site_color_schemes.columns:
        print(f"Using the {color_scheme} site color scheme")
        site_colors = site_color_schemes.set_index('site')[color_scheme].to_dict()
        df = df.assign(color=lambda x: x['site'].map(site_colors))
        dists = []
        colors = []
        for condition, condition_df in (
                df
                [['condition', 'color', 'site', site_metric]]
                .drop_duplicates()
                .assign(site_metric2=lambda x: x[site_metric]**2)  # color in proportion to **square** of site escape
                .groupby(['condition', 'color'])
                .aggregate(tot_escape=pd.NamedAgg('site_metric2', 'sum'))
                .reset_index()
                .sort_values('tot_escape', ascending=False)
                .assign(condition=lambda x: pd.Categorical(x['condition'], conditions, ordered=True))
                .groupby('condition', sort=True)
                ):
            dists.append(condition_df['tot_escape'].tolist())
            colors.append(condition_df['color'].tolist())
    else:
        print(f"Coloring all points {color_scheme}")
        dists = [[1] for conditition in conditions]
        colors = [[color_scheme] for condition in conditions]
        
    # get circle / label colors
    if 'default_circle_color' in specs:
        default_circle_color = specs['default_circle_color']
    else:
        default_circle_color = 'none'
    if 'default_label_color' in specs:
        default_label_color = specs['default_label_color']
    else:
        default_label_color = 'black'
    circle_colors = []
    label_colors = []
    for condition in conditions:
        if 'circle_colors' in specs and condition in specs['circle_colors']:
            circle_colors.append(specs['circle_colors'][condition])
        else:
            circle_colors.append(default_circle_color)
        if 'label_colors' in specs and condition in specs['label_colors']:
            label_colors.append(specs['label_colors'][condition])
        else:
            label_colors.append(default_label_color)
    
    # plot the multidimensional scaling result
    plot_size = 4 if 'plot_size' not in specs else specs['plot_size']
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    xs = locs[:, 0]
    ys = locs[:, 1]
    for x, y, dist, color, circle_color in zip(xs, ys, dists, colors, circle_colors):
        draw_pie(dist, x, y,
                 size=300 if 'pie_size' not in specs else specs['pie_size'],
                 ax=ax,
                 colors=color,
                 alpha=0.7 if 'pie_alpha' not in specs else specs['pie_alpha'],
                 circle_color=circle_color,
                 )
    ax.set_aspect('equal', adjustable='box')  # same distance on both axes
    ax.set_xticks([])  # no x-ticks
    ax.set_yticks([])  # no y-ticks
    ax.margins(0.09)  # increase padding from axes
    if 'no_labels' not in specs or not specs['no_labels']:
        texts = [plt.text(x, y, label, color=color) for x, y, label, color
                 in zip(xs, ys, conditions, label_colors)]
        adjustText.adjust_text(texts,
                               x=xs,
                               y=ys,
                               expand_points=(1.2, 1.6) if 'expand_points' not in specs
                                             else specs['expand_points'],
                               )
    plotfile = os.path.join(config['mds_dir'], f"{name}_mds.pdf")
    print(f"Saving plot to {plotfile}")
    fig.savefig(plotfile, bbox_inches='tight')
    plt.show(fig)
    plt.close(fig)
```

    
    Making plot all, which has the following antibodies:
    ['P02_500', 'P03_1250', 'P04_1250', 'P05_500', 'P08_500', 'P09_200', 'P12_200', 'P14_1250', 'M02-day-119_200', 'M05-day-119_500', 'M08-day-119_200', 'M10-day-119_200', 'M12-day-119_200', 'M16-day-119_1250', 'M17-day-119_200', 'M18-day-119_80', 'M19-day-119_200', 'M01-day-119_80', 'M03-day-119_200', 'M04-day-119_200', 'M06-day-119_80', 'M07-day-119_200', 'M09-day-119_500', 'M11-day-119_200', 'M13-day-119_200', 'M14-day-119_500', 'M20-day-119_200', 'M21-day-119_200', 'M22-day-119_200', 'M23-day-119_200', '12C_d61_160', '13_d15_200', '1C_d26_200', '22C_d28_200', '23C_d26_80', '23_d21_1250', '24C_d32_200', '25C_d48_200', '25_d18_500', '6C_d33_500', '7C_d29_500', 'COV-021_500', 'COV-047_200', 'COV-057_50', 'COV-072_200', 'COV-107_80', 'C105_400', 'CB6_400', 'COV2-2165_400', 'COV2-2196_400', 'COV2-2832_400', 'REGN10933_400', 'C002_400', 'C121_400', 'C144_400', 'COV2-2479_400', 'LY-CoV555_400', 'COV2-2050_400', 'C110_400', 'C135_400', 'COV2-2096_400', 'COV2-2130_400', 'COV2-2499_400', 'REGN10987_400', 'COV2-2082_400', 'COV2-2094_400', 'COV2-2677_400', 'CR3022_400', '267C_200', '268C_500', '273C_500', '274C_500', '276C_500', '277C_500', '278C_1250', '279C_1250', 'Delta_1_500', 'Delta_3_350', 'Delta_4_350', 'Delta_6_500', 'Delta_7_1250', 'Delta_8_500', 'Delta_10_1250', 'Delta_11_500', 'K007_500', 'K031_500', 'K033_500', 'K040_500', 'K041_500', 'K046_200', 'K114_200', 'K115_old_80', 'K119_200']
    stress = 63.80033988120528 from iteration 144
    Manual calculus of sklearn stress : 63.80020748657694
    Kruskal's Stress : 0.20304026742143952
    [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
    Using condition-level color scheme in column class_color of data/mds_color_schemes.csv
    Saving plot to results/multidimensional_scaling/all_mds.pdf



    
![png](mds_escape_profiles_files/mds_escape_profiles_18_1.png)
    


    
    Making plot primary_infections, which has the following antibodies:
    ['12C_d61_160', '13_d15_200', '1C_d26_200', '22C_d28_200', '23C_d26_80', '23_d21_1250', '24C_d32_200', '25C_d48_200', '25_d18_500', '6C_d33_500', '7C_d29_500', 'COV-021_500', 'COV-047_200', 'COV-057_50', 'COV-072_200', 'COV-107_80', 'C105_400', 'CB6_400', 'COV2-2165_400', 'COV2-2196_400', 'COV2-2832_400', 'REGN10933_400', 'C002_400', 'C121_400', 'C144_400', 'COV2-2479_400', 'LY-CoV555_400', 'COV2-2050_400', 'C110_400', 'C135_400', 'COV2-2096_400', 'COV2-2130_400', 'COV2-2499_400', 'REGN10987_400', 'COV2-2082_400', 'COV2-2094_400', 'COV2-2677_400', 'CR3022_400', 'Delta_1_500', 'Delta_3_350', 'Delta_4_350', 'Delta_6_500', 'Delta_7_1250', 'Delta_8_500', 'Delta_10_1250', 'Delta_11_500', 'K007_500', 'K031_500', 'K033_500', 'K040_500', 'K041_500', 'K046_200', 'K114_200', 'K115_old_80', 'K119_200']
    stress = 27.06344402911881 from iteration 156
    Manual calculus of sklearn stress : 27.06340618796206
    Kruskal's Stress : 0.20342145923343238
    [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
    Using condition-level color scheme in column class_color of data/mds_color_schemes.csv
    Saving plot to results/multidimensional_scaling/primary_infections_mds.pdf



    
![png](mds_escape_profiles_files/mds_escape_profiles_18_3.png)
    


    
    Making plot Delta_vs_breakthrough, which has the following antibodies:
    ['C105_400', 'CB6_400', 'COV2-2165_400', 'COV2-2196_400', 'COV2-2832_400', 'REGN10933_400', 'C002_400', 'C121_400', 'C144_400', 'COV2-2479_400', 'LY-CoV555_400', 'COV2-2050_400', 'C110_400', 'C135_400', 'COV2-2096_400', 'COV2-2130_400', 'COV2-2499_400', 'REGN10987_400', 'COV2-2082_400', 'COV2-2094_400', 'COV2-2677_400', 'CR3022_400', '267C_200', '268C_500', '273C_500', '274C_500', '276C_500', '277C_500', '278C_1250', '279C_1250', 'Delta_1_500', 'Delta_3_350', 'Delta_4_350', 'Delta_6_500', 'Delta_7_1250', 'Delta_8_500', 'Delta_10_1250', 'Delta_11_500']
    stress = 11.939765275965488 from iteration 141
    Manual calculus of sklearn stress : 11.938841658603192
    Kruskal's Stress : 0.1878279851266539
    [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
    Using condition-level color scheme in column class_color of data/mds_color_schemes.csv
    Saving plot to results/multidimensional_scaling/Delta_vs_breakthrough_mds.pdf



    
![png](mds_escape_profiles_files/mds_escape_profiles_18_5.png)
    


    
    Making plot primaries_breakthrough, which has the following antibodies:
    ['12C_d61_160', '13_d15_200', '1C_d26_200', '22C_d28_200', '23C_d26_80', '23_d21_1250', '24C_d32_200', '25C_d48_200', '25_d18_500', '6C_d33_500', '7C_d29_500', 'COV-021_500', 'COV-047_200', 'COV-057_50', 'COV-072_200', 'COV-107_80', 'C105_400', 'CB6_400', 'COV2-2165_400', 'COV2-2196_400', 'COV2-2832_400', 'REGN10933_400', 'C002_400', 'C121_400', 'C144_400', 'COV2-2479_400', 'LY-CoV555_400', 'COV2-2050_400', 'C110_400', 'C135_400', 'COV2-2096_400', 'COV2-2130_400', 'COV2-2499_400', 'REGN10987_400', 'COV2-2082_400', 'COV2-2094_400', 'COV2-2677_400', 'CR3022_400', '267C_200', '268C_500', '273C_500', '274C_500', '276C_500', '277C_500', '278C_1250', '279C_1250', 'Delta_1_500', 'Delta_3_350', 'Delta_4_350', 'Delta_6_500', 'Delta_7_1250', 'Delta_8_500', 'Delta_10_1250', 'Delta_11_500', 'K007_500', 'K031_500', 'K033_500', 'K040_500', 'K041_500', 'K046_200', 'K114_200', 'K115_old_80', 'K119_200']
    stress = 31.089955021533697 from iteration 270
    Manual calculus of sklearn stress : 31.089347121980364
    Kruskal's Stress : 0.20054844935371646
    [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
    Using condition-level color scheme in column class_color of data/mds_color_schemes.csv
    Saving plot to results/multidimensional_scaling/primaries_breakthrough_mds.pdf



    
![png](mds_escape_profiles_files/mds_escape_profiles_18_7.png)
    



```python

```
