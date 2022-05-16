### This repo is created to track status of work on pyprobml notebooks and make review process easy using review-nb bot. Please follow below steps:

1. First create an issue before stating any figure. Issue title can be like this `Fig2.10 (book1) gauss_2d_plot.ipyb`
2. Then fork this repo and create a new branch with meaningful name.
3. After your notebook is ready for review, create a PR (with same title of issue) on this repo so we can review/discuss them and add comments in PR itself. Before opening a PR, make sure it is `pyprobml-review` repo and not the original `pyprobml` repo. We will soon shift to the original repo! 
4. You should add latexified figures in your PR (see this [gist](https://gist.github.com/karm-patel/15b1e1895756088725872bba9204c9d1) for example). You can use overleaf latex template to render latexified figures
6. Once confirmed by one of the reviewers, you can create a PR on main repo.

### [[Important] Follow these instructions carefully](https://github.com/probml/pyprobml/tree/master/notebooks#notebooks)

## UPDATE
There are some changes which we need to keep in mind and update code accordingly. see this [reference notebook](https://github.com/probml/pyprobml/blob/master/notebooks/book1/02/discrete_prob_dist_plot.ipynb) to compare these changes.
1. We do not need to add try...except when package is available in [requirement.txt](https://github.com/probml/pyprobml/blob/master/requirements.txt) 
2. Now [probml-utils](https://github.com/probml/probml-utils) became local installable pip package. path of probml-utils is given in [requirement.txt](https://github.com/probml/pyprobml/blob/master/requirements.txt)
3. The way of working of `latexify()` is slighly changed, basically we do not need to put it under condition `if LATEXIFY`. I recommend to see `plotting.py` file in probml-utils repo to understand these changes.
4. In orrder to save figure we need to save path in `FIG_DIR` environment variable. if this path is not set, code will run but it will not save any figure.
