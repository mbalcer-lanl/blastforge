Blast Forge: Deep RL-based Explosive Design
===========================================

![Blow it up!](./blastforge_logo.png)


About:
------

A set of tools to enable Deep Reinforcement Learning for explosive
device design.

`BlastForge` is built off of `torchrl`. 


Installation:
-------------

The python environment is specified through the `pyproject.toml`
file. `BlastForge` is meant to be installed using `flit` in a minimal
python environment.

Setup your base environment and activate it (we use conda):

```
>> conda create -n <blastforge_env_name> python=3.9 flit
>> conda activate <blastforge_env_name>
```

> **WARNING!!**
>
> For some environments `flit`, the install manager for `BlastForge` will not
> default to installing in the conda environment. To remedy this first
> checkout your `USER_BASE` and `USER_SITE` variables using
>
> ```
> >> python -m site
> ```
>
> If `USER_BASE` and `USER_SITE` don't appear to be associated with
> `<blastforge_env_name>` then set the `PYTHONUSERBASE` environment variable
> prior to installing `BlastForge`:
>
> ```
> >> export PYTHONUSERBASE=$CONDA_PREFIX
> ```
>
> Rerun `python -m site` to ensure `USER_BASE` and `USER_SITE` have
> changed.

For **developers**, you can install a **development version** of
`BlastForge` checkout using...

```
>> flit install --user --symlink --deps=all
```

For **non-developers**, you can install `BlastForge` using...

```
>> flit install --deps=all
```

> **WARNING**
> 
> This install process does not guarantee that PyTorch is installed to
> utilize your GPUs. If you want to ensure that PyTorch is installed to
> make optimal use of your hardware we suggest manually installing
> `torch` prior to installing `BlastForge` with `flit`.


Testing:
--------

To run the tests use...

```
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```


Linting:
--------

The `ruff` linter is used in `BlastForge` to enforce coding and formatting
standards. To run the linter do

```
>> ruff check
>> ruff check --preview
```

You can make `ruff` fix automatic standards using

```
>> ruff check --fix
>> ruff check --preview --fix
```

Use `ruff` to then check your code formatting and show you what would
be adjusted, then fix formatting

```
>> ruff format --check --diff
>> ruff format
```


Copyright:
----------

LANL **O4864**

&copy; 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los
Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for
the U.S. Department of Energy/National Nuclear Security Administration. All rights in
the program are reserved by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government is granted for itself
and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others to do so.