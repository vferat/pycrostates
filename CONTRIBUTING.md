Contributing to Pycrostates
==========================

Once the contribution discussed, you can propose a change by creating a new pull request:

- [Fork the Pycrostates Repository](https://github.com/vferat/pycrostates/fork) on Github.

- Clone your forked repository

- We recommand to create a new branch on which you will make your changes

- We recommand to create a new environment, for exemple with Anaconda:
    ```console
    conda create -n pycrostates_dev python=3.9
    ```
    then activate your environment:

    ```console
    conda activate -n pycrostates_dev
    ```

- Navigate to the repository's root folder

- Install pycrostates in editable mode:
    ```console
    pip install -e .[all]
    ```

- Make your changes


## Code style

Pycrostates uses [flake](https://github.com/PyCQA/flake8), [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) to enforce code style.

For exemple, form the pycrostates root directory you can run:
- flake8
    ```console
    flake8 pycrostates
    ```

- black
    ```console
    black pycrostates
    ```

- isort
    ```console
    isort pycrostates
    ```

## Running the test suite

```console
pytest pycrostates
```

## Building the documentation

From within the `/docs` directroy:
```console
make html
```