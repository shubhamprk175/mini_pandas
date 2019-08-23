# About Mini Pandas

This repository contains a Data Analysis Library built with Python, namely mini_pandas. It is a fully-functioning library similar to pandas.


Most data scientists who use Python rely on pandas. This project, mini pandas, is a library that implements many of the most common and useful methods found in pandas. Mini Pandas will:

* Have a DataFrame class with data stored in numpy arrays
* Select subsets of data with the brackets operator
* Use special methods defined in the Python data model
* Have a nicely formatted display of the DataFrame in the notebook
* Implement aggregation methods - sum, min, max, mean, median, etc...
* Implement non-aggregation methods such as isna, unique, rename, drop
* Group by one or two columns
* Have methods specific to string columns
* Read in data from a comma-separated value file


## Setting up the Development Environment

I recommend creating a new environment using the conda package manager. If you do not have conda, you can [download it here][0] along with the entire Anaconda distribution. Choose Python 3. When beginning development on a new library, it's a good idea to use a completely separate environment to write your code.

### Create the environment with the `environment.yml` file

Conda allows you to automate the environment creation with an `environment.yml` file. The contents of the file are minimal and are displayed below.

```yml
name: mini_pandas
dependencies:
- python=3.6
- pandas
- jupyter
- pytest
```

This file will be used to create a new environment named `mini_pandas`. It will install Python 3.6 in a completely separate directory in your file system along with pandas, jupyter, and pytest. There will actually be many more packages installed as those libraries have dependencies of their own. Visit [this page][2] for more information on conda environments.

### Command to create new environment

In the top level directory of this repository, where the `environment.yml` file is located, run the following from your command line.

`conda env create -f environment.yml`

The above command will take some time to complete. Once it completes, the environment will be created.
