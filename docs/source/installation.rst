Installation
============

To install **ConnInfPy**, use one of the approaches below.


1. Install the latest stable version (includes JIT speedup):

.. code-block:: bash

    pip install conninfpy

---------------

2. Clone this repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/IHB-IBR-department/ConnInfPy.git
    cd ConnInfPy
    pip install -e .

---------------

3. Install with development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

---------------

Requirements:
---------------
- Python **>=3.8** (tested on 3.11)
- Required: ``numpy``, ``scipy``, ``statsmodels``, ``pandas``, ``matplotlib``, ``numba``
- Optional ``[full]`` extra: ``scikit-learn``, ``pyyaml``, ``networkx``
- Optional ``[dev]`` extra: ``pytest``, ``sphinx``, ``sphinx-rtd-theme``,
  ``myst-parser``, ``seaborn``, ``mypy``

.. note::
   **Safe Install:** If the default installation fails (usually due to ``numba`` or ``llvmlite``
   compilation issues on legacy systems), you can install without dependencies:

   .. code-block:: bash

       pip install conninfpy --no-deps
       pip install numpy scipy statsmodels pandas matplotlib

   The library will automatically detect the missing ``numba`` and fall back to the SciPy backend.
