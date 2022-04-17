### This repo is created to track status of work on pyprobml notebooks and make review process easy using review-nb bot. Please follow below steps:

1. First create an issue before stating any figure. Issue title can be like this `Fig2.10 (book1) gauss_2d_plot.ipyb`
2. Then fork this repo and create a new branch with meaningful name.
3. After your notebook is ready for review, create a PR (with same title of issue) on this repo so we can review/discuss them and add comments in PR itself. 
4. You should add latexified figures in your PR (see this [gist](https://gist.github.com/karm-patel/15b1e1895756088725872bba9204c9d1) for example). You can use overleaf latex template to render latexified figures
6. Once reviewed, you can create a PR on main repo.


### Common instructions

* Wrap your imports with `try: except:` but in following cases, do not wrap them:
    * If a module is in the [requirements.txt](https://github.com/probml/pyprobml/blob/master/requirements.txt)
    * If a module is an inbuilt module of Python, such as `os`, `sys` etc.
* Import latexify and savefig functions from [probml_utils](https://github.com/probml/probml-utils) repo. In case, you find any difficulty in using this, or find any bug, open an issue on this repo (pyprobml-review repo) stating the details.

### Recent major changes in practice
* Use both `FIG_DIR` and `LATEXIFY` variables to generate latexified figures. `FIG_DIR` should contain the location where you would like to save the figures. For example:
```py
FIG_DIR=/home/patel_zeel/figures/ LATEXIFY=1 foo.ipynb
```
