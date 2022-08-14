Contributing to Pycrostates
==========================

Once the contribution discussed, you can propose a change by creating a new pull request. Pycrostates tries to adhere as much as possible to the conventions used by MNE-python, so we recommend reading the MNE-python [contribution guide](https://mne.tools/dev/install/contributing.html) for more details about this process.

- [Fork the Pycrostates Repository](https://github.com/vferat/pycrostates/fork) on Github.

- Clone your forked repository locally.

- [OPTIONAL] create a new branch on which you will add your changes.

- [OPTIONAL] create a new environment, for example with Anaconda:

    ```console
    conda create -n pycrostates_dev python=3.9
    ```

    then activate your environment:

    ```console
    conda activate -n pycrostates_dev
    ```

- Navigate to the repository's root folder

- Install pycrostates in editable mode with all optional dependencies:

    ```console
    pip install -e .[all]
    ```

- Make your changes.

- [Open a pull request](https://github.com/vferat/pycrostates/compare)

## Code style

Pycrostates uses [flake](https://github.com/PyCQA/flake8), [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) to enforce code style.

You can run these tools from the pycrostates root directory using the corresponding command:

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

From within the `/docs` directory:

```console
make html
```
