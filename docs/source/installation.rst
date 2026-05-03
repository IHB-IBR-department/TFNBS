Installation
============

To install **ConnInfPy**, use one of the approaches below.


1. Clone this repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/IHB-IBR-department/ConnInfPy.git
    cd ConnInfPy
    pip install -e .

---------------

2. Install with development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

---------------

3. Install with optional fast backend:

.. code-block:: bash

    pip install -e ".[fast]"          # adds numba (8–31× JIT speedup on TFNBS scoring)

---------------

Requirements:
---------------
- Python **>=3.8** (tested on 3.11)
- Required: ``numpy``, ``scipy``, ``statsmodels``, ``matplotlib``,
  ``scikit-learn``, ``networkx``, ``pandas``, ``pyyaml``
- Optional ``[fast]`` extra: ``numba`` (8–31× JIT enhancement backend)
- Optional ``[dev]`` extra: ``pytest``, ``sphinx``, ``sphinx-rtd-theme``,
  ``myst-parser``, ``seaborn``
- Optional ``[notebooks]`` extra: ``jupyter``, ``ipykernel``,
  ``matplotlib``, ``seaborn``
