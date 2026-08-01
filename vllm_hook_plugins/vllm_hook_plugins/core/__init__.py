"""Engine-free core of the declarative steering/capture surface.

Everything importable from here runs without vLLM: kind registries, spec
parsing/validation, canonical hashing, the reference interpreter math,
the content-addressed artifact registry, and fingerprint recipes. The
unified worker calls into this package; external tools can import it to
validate specs, derive cache salts, and reproduce interventions offline.
"""

__version__ = "0.3.0"
