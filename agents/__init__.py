from agents.crl import CRLAgent
from agents.dsharsa import DSHARSAAgent
from agents.fdrl import FDRLAgent
from agents.gcfbc import GCFBCAgent
from agents.gcfql import GCFQLAgent
from agents.gciql import GCIQLAgent
from agents.gcsacbc import GCSACBCAgent
from agents.hfdrl import HFDRLAgent
from agents.hgcfbc import HGCFBCAgent
from agents.hiql import HIQLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.qrl import QRLAgent
from agents.sharsa import SHARSAAgent

agents = dict(
    crl=CRLAgent,
    dsharsa=DSHARSAAgent,
    fdrl=FDRLAgent,
    gcfbc=GCFBCAgent,
    gcfql=GCFQLAgent,
    gciql=GCIQLAgent,
    gcsacbc=GCSACBCAgent,
    hfdrl=HFDRLAgent,
    hgcfbc=HGCFBCAgent,
    hiql=HIQLAgent,
    ngcsacbc=NGCSACBCAgent,
    qrl=QRLAgent,
    sharsa=SHARSAAgent,
)
