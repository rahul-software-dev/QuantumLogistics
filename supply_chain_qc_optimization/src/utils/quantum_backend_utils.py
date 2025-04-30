"""
Quantum Backend Utilities
------------------------
Manage quantum hardware and simulator settings for reproducible experiments.
References:
- Qiskit Documentation: https://qiskit.org/documentation/
"""
from qiskit import Aer, IBMQ

def get_backend(backend_name='aer_simulator_statevector', use_ibmq=False, ibmq_token=None, hub=None, group=None, project=None):
    """
    Returns a Qiskit backend (simulator or real device).
    """
    if use_ibmq:
        if not IBMQ.active_account():
            IBMQ.enable_account(ibmq_token)
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        backend = provider.get_backend(backend_name)
    else:
        backend = Aer.get_backend(backend_name)
    return backend

def list_available_backends(use_ibmq=False, ibmq_token=None, hub=None, group=None, project=None):
    """
    Lists available backends.
    """
    if use_ibmq:
        if not IBMQ.active_account():
            IBMQ.enable_account(ibmq_token)
        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        return provider.backends()
    else:
        return Aer.backends()
