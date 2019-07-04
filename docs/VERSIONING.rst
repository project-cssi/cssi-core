VERSIONING CONVENTION
=====================

Please follow the following `PEP0440`_ release version standard.

.. _PEP0440: https://www.python.org/dev/peps/pep-0440/

Final Releases
--------------

.. code-block:: bash

  X(.Y)*

  Ex: 0.9
    0.9.1
    0.9.2

Pre-releases
------------

.. code-block:: bash

  X.YaN   # Alpha release
  X.YbN   # Beta release
  X.YrcN  # Release Candidate
  X.Y.Z   # For bug-fix releases

Post-releases
-------------

.. code-block:: bash

  X.YaN.postM   # Post-release of an alpha release
  X.YbN.postM   # Post-release of a beta release
  X.YrcN.postM  # Post-release of a release candidate

Developmental releases
----------------------

.. code-block:: bash

  X.YaN.devM       # Developmental release of an alpha release
  X.YbN.devM       # Developmental release of a beta release
  X.YrcN.devM      # Developmental release of a release candidate
  X.Y.postN.devM   # Developmental release of a post-release
