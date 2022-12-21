from typing import Any
from ampel.base.LogicalUnit import LogicalUnit
from ampel.core.AmpelContext import AmpelContext
from ampel.model.UnitModel import UnitModel
from ampel.protocol.LoggerProtocol import LoggerProtocol
from ampel.struct.T3Store import T3Store
import pytest
from pathlib import Path
import yaml
import logging
import requests

from ampel.ztf.t3.complement.TNSNames import TNSNames
from ampel.ztf.t3.complement.TNSReports import TNSReports
from ampel.ztf.t2.T2CatalogMatch import T2CatalogMatch
from ampel.ztf.t0.DecentFilter import DecentFilter

from ampel.content.DataPoint import DataPoint
from ampel.content.StockDocument import StockDocument
from ampel.content.T2Document import T2Document

from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.log.AmpelLogger import AmpelLogger
from ampel.enum.DocumentCode import DocumentCode


@pytest.fixture
def catalogmatch_config():
    with open(Path(__file__).parent / "test-data" / "catalogmatch_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def catalogmatch_service_reachable():
    try:
        requests.head("https://ampel.zeuthen.desy.de/", timeout=0.5)
    except requests.exceptions.Timeout:
        pytest.skip("https://ampel.zeuthen.desy.de/ is unreachable")


@pytest.fixture
def ampel_logger():
    return AmpelLogger.get_logger()


def test_catalogmatch(
    patch_mongo,
    dev_context: AmpelContext,
    catalogmatch_config: dict[str, Any],
    ampel_logger: AmpelLogger,
    catalogmatch_service_reachable,
):
    unit: T2CatalogMatch = dev_context.loader.new_logical_unit(
        model=UnitModel(unit="T2CatalogMatch", config=catalogmatch_config),
        logger=ampel_logger,
        sub_type=T2CatalogMatch,
    )
    result = unit.process(DataPoint({"id": 0, "body": {"ra": 0, "dec": 0}}))
    base_config: dict[str, Any] = {
        k: None for k in catalogmatch_config["catalogs"].keys()
    }
    assert result == (
        base_config
        | {
            "SDSS_spec": {
                "bptclass": 5.0,
                "dist2transient": 0.0,
                "subclass": "n/a",
                "z": 0.06367020308971405,
            },
            "GLADEv23": {
                "dist2transient": 0.0,
                "flag1": "G",
                "dist": None,
                "dist_err": None,
                "z": None,
                "flag2": 0,
                "flag3": 0,
            },
        }
    )


def test_decentfilter_star_in_gaia(
    patch_mongo,
    dev_context: AmpelContext,
    ampel_logger: AmpelLogger,
):
    with open(Path(__file__).parent / "test-data" / "decentfilter_config.yaml") as f:
        config = yaml.safe_load(f)
    unit: DecentFilter = dev_context.loader.new_logical_unit(
        UnitModel(unit="DecentFilter", config=config),
        logger=ampel_logger,
        sub_type=DecentFilter,
    )
    assert unit.is_star_in_gaia(
        {"ra": 0.009437700971970959, "dec": -0.0008937364197194631}
    )
    assert not unit.is_star_in_gaia({"ra": 0, "dec": 0})


def test_tnsnames(
    patch_mongo, dev_context: AmpelContext, ampel_logger: AmpelLogger
) -> None:
    unit: TNSNames = dev_context.loader.new_context_unit(
        UnitModel(unit="TNSNames"),
        logger=ampel_logger,
        context=dev_context,
        sub_type=TNSNames,
    )
    buf = AmpelBuffer(
        {
            "id": 0,
            "stock": StockDocument(
                {
                    "stock": 0,
                    "tag": [],
                    "channel": [],
                    "journal": [],
                    "ts": {},
                    "updated": 0,
                    "name": ["sourceysource"],
                }
            ),
            "t2": [
                T2Document(
                    {
                        "unit": "T2LightCurveSummary",
                        "meta": [{"code": DocumentCode.OK, "tier": 2}, {"tier": 0}],
                        "body": [{"ra": 0.518164, "dec": 0.361964}],
                    }
                )
            ],
        }
    )
    unit.complement([buf], T3Store())
    assert (stockdoc := buf["stock"]) is not None
    assert stockdoc["name"] == ("sourceysource", "TNS2020ubb")
    assert not "extra" in buf

    unit = dev_context.loader.new_context_unit(
        UnitModel(unit="TNSReports"),
        logger=ampel_logger,
        context=dev_context,
        sub_type=TNSNames,
    )
    unit.complement([buf], T3Store())
    assert stockdoc["name"] == ("sourceysource", "TNS2020ubb")
    assert buf["extra"] == {
        "TNSReports": [
            {
                "decdeg_err": 2.5e-05,
                "discmagfilter": {"id": 111, "name": "r", "family": "ZTF"},
                "discoverer": "F. Forster, A. Munoz-Arancibia, F.E. Bauer, L. Hernandez-Garcia, L. Galbany, G. Pignata, E. Camacho, J. Silva-Farfan, A. Mourao, J. Arredondo, G. Cabrera-Vives, R. Carrasco-Davis, P.A. Estevez, P. Huijse, E. Reyes, I. Reyes, P. Sanchez-Saez, C. Valenzuela, E. Castillo, D. Ruz-Mieres, D. Rodriguez-Mancini, M. Catelan, S. Eyheramendy, M.J. Graham on behalf of the ALeRCE broker",
                "discoverer_internal_name": "ZTF20acddzrd",
                "discovery_data_source": {"groupid": 48, "group_name": "ZTF"},
                "discoverydate": "2020-09-23T08:28:24.004000",
                "discoverymag": 19.1521,
                "end_prop_period": None,
                "host_redshift": 0.301422,
                "hostname": "WISEA J000204.63+002140.7",
                "internal_names": "ZTF20acddzrd",
                "name_prefix": "AT",
                "object_type": {"name": None, "id": None},
                "objid": 67022,
                "objname": "2020ubb",
                "public": 1,
                "radeg_err": 2.5e-05,
                "redshift": None,
                "reporter": "ALeRCE",
                "reporterid": 66140,
                "reporting_group": {"groupid": 74, "group_name": "ALeRCE"},
                "source": "bot",
            }
        ]
    }
