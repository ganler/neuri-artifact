import logging

VIZ_LOG = logging.getLogger("viz")
FUZZ_LOG = logging.getLogger("fuzz")
MGEN_LOG = logging.getLogger("mgen")
SMT_LOG = logging.getLogger("smt")
EXEC_LOG = logging.getLogger("exec")
DTEST_LOG = logging.getLogger("dtest")
CORE_LOG = logging.getLogger("core")
AUTOINF_LOG = logging.getLogger("autoinf")

TF_LOG = logging.getLogger("gen|tf")
TORCH_LOG = logging.getLogger("gen|torch")

AUTOINF_LOG.setLevel(logging.INFO)
