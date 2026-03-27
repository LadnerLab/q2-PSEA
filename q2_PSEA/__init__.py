#! /usr/bin/env python
from q2_PSEA.actions.psea import (
    make_psea_table,
    run_iterative_peptide_analysis,
    run_iterative_process_single_pair,
    create_fgsea_table_for_pair,
)

__all__ = [
    "make_psea_table",
    "run_iterative_peptide_analysis",
    "run_iterative_process_single_pair",
    "create_fgsea_table_for_pair",
]

from . import _version
__version__ = _version.get_versions()["version"]