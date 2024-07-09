import argparse
import logging
import os
import pathlib
import pickle
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import colorlog
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

DATA_DIR = str(pathlib.Path("~/data").expanduser())
os.environ["DATA_DIR"] = DATA_DIR


def set_logger_config(
    level: int = logging.INFO,
    log_file: Optional[Union[str, pathlib.Path]] = None,
    log_to_stdout: bool = True,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once. Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level : int
        The default level for logging. Default is ``logging.INFO``.
    log_file : str | Path, optional
        The file to save the log to.
    log_to_stdout : bool
        A flag indicating if the log should be printed on stdout. Default is True.
    colors : bool
        A flag for using colored logs. Default is True.
    """
    base_logger = logging.getLogger("ENTSO-E Downloader")
    simple_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if colors:
        formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(formatter)
    else:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(simple_formatter)

    if log_to_stdout:
        base_logger.addHandler(std_handler)
    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)
    base_logger.setLevel(level)


log = logging.getLogger("ENTSO-E Downloader")  # Get logger instance.
set_logger_config()

# selected from the Enum in
# https://github.com/EnergieID/entsoe-py/blob/4560a5d7d96d964f5cf50370d576662f190fd500/entsoe/mappings.py
# and from the ENTSO-E transparency platform for loads
# and contains the bidding zones from below
ALL_BIDDING_ZONES = [
    "AL",
    "AM",
    "AT",
    "AZ",
    "BA",
    "BE",
    "BG",
    "BY",
    "CH",
    "CY",
    "CZ",
    "CZ_DE_SK",
    "DE_AT_LU",
    "DE_LU",
    "DK_1",
    "DK_1_NO_1",
    "DK_2",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GB_ELECLINK",
    "GB_IFA",
    "GB_IFA2",
    "GE",
    "GR",
    "HR",
    "HU",
    "IE_SEM",
    "IT_BRNN",
    "IT_CALA",
    "IT_CNOR",
    "IT_CSUD",
    "IT_FOGN",
    "IT_GR",
    "IT_MALTA",
    "IT_NORD",
    "IT_NORD_AT",
    "IT_NORD_CH",
    "IT_NORD_FR",
    "IT_NORD_SI",
    "IT_PRGP",
    "IT_ROSN",
    "IT_SACO_AC",
    "IT_SACO_DC",
    "IT_SARD",
    "IT_SICI",
    "IT_SUD",
    "LT",
    "LU_BZN",
    "LV",
    "MD",
    "ME",
    "MK",
    "MT",
    "NL",
    "NO_1",
    "NO_1A",
    "NO_2",
    "NO_2A",
    "NO_2_NSL",
    "NO_3",
    "NO_4",
    "NO_5",
    "PL",
    "PT",
    "RO",
    "RS",
    "RU",
    "RU_KGD",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
    "SI",
    "SK",
    "TR",
    "UA",
    "UA_BEI",
    "UA_DOBTPP",
    "UA_IPS",
    "XK",
]

# symmetric dictionary of neighbours.
# contains all edge information from entsoe.mappings.NEIGHBOURS aside from "IT" as it is not a bidding zone
# contains all neighbour information from the entsoe web interface
NEIGHBORS = {
    "AL": ["GR", "ME", "RS", "XK"],
    "AM": ["GE"],
    "AT": ["CH", "CZ", "DE_LU", "HU", "IT_NORD", "SI"],
    "AZ": ["GE"],
    "BA": ["HR", "ME", "RS"],
    "BE": ["DE_AT_LU", "DE_LU", "FR", "GB", "NL"],
    "BG": ["GR", "MK", "RO", "RS", "TR"],
    "BY": ["LT", "UA", "UA_IPS"],
    "CH": ["AT", "DE_AT_LU", "DE_LU", "FR", "IT_NORD", "IT_NORD_CH"],
    "CZ": ["AT", "DE_AT_LU", "DE_LU", "PL", "SK"],
    "CZ_DE_SK": ["PL"],
    "DE_AT_LU": [
        "BE",
        "CH",
        "CZ",
        "DK_1",
        "DK_2",
        "FR",
        "HU",
        "IT_NORD",
        "IT_NORD_AT",
        "NL",
        "PL",
        "SE_4",
        "SI",
    ],
    "DE_LU": ["AT", "BE", "CH", "CZ", "DK_1", "DK_2", "FR", "NL", "NO_2", "PL", "SE_4"],
    "DK_1": ["DE_AT_LU", "DE_LU", "DK_2", "GB", "NL", "NO_2", "SE_3", "SE_4"],
    "DK_1_NO_1": ["SE_3"],
    "DK_2": ["DE_AT_LU", "DE_LU", "DK_1", "SE_4"],
    "EE": ["FI", "LV", "RU"],
    "ES": ["FR", "PT"],
    "FI": ["EE", "NO_4", "RU", "SE_1", "SE_3"],
    "FR": [
        "BE",
        "CH",
        "DE_AT_LU",
        "DE_LU",
        "ES",
        "GB",
        "GB_ELECLINK",
        "GB_IFA",
        "GB_IFA2",
        "IT_NORD",
        "IT_NORD_FR",
    ],
    "GB": [
        "BE",
        "DK_1",
        "FR",
        "GB_ELECLINK",
        "GB_IFA",
        "GB_IFA2",
        "IE_SEM",
        "NL",
        "NO_2",
    ],
    "GB_ELECLINK": ["FR", "GB"],
    "GB_IFA": ["FR", "GB"],
    "GB_IFA2": ["FR", "GB"],
    "GE": ["AM", "AZ", "RU", "TR"],
    "GR": ["AL", "BG", "IT_BRNN", "IT_GR", "IT_SUD", "MK", "TR"],
    "HR": ["BA", "HU", "RS", "SI"],
    "HU": ["AT", "DE_AT_LU", "HR", "RO", "RS", "SI", "SK", "UA", "UA_BEI", "UA_IPS"],
    "IE_SEM": ["GB"],
    "IT_BRNN": ["GR", "IT_SUD"],
    "IT_CALA": ["IT_SICI", "IT_SUD"],
    "IT_CNOR": ["IT_CSUD", "IT_NORD", "IT_SACO_DC", "IT_SARD"],
    "IT_CSUD": ["IT_CNOR", "IT_SARD", "IT_SUD", "ME"],
    "IT_FOGN": ["IT_SUD"],
    "IT_GR": ["GR"],
    "IT_MALTA": ["MT"],
    "IT_NORD": ["AT", "CH", "DE_AT_LU", "FR", "IT_CNOR", "SI"],
    "IT_NORD_AT": ["DE_AT_LU"],
    "IT_NORD_CH": ["CH"],
    "IT_NORD_FR": ["FR"],
    "IT_NORD_SI": ["SI"],
    "IT_PRGP": ["IT_SICI"],
    "IT_ROSN": ["IT_SICI", "IT_SUD"],
    "IT_SACO_AC": ["IT_SARD"],
    "IT_SACO_DC": ["IT_CNOR", "IT_SARD"],
    "IT_SARD": ["IT_CNOR", "IT_CSUD", "IT_SACO_AC", "IT_SACO_DC"],
    "IT_SICI": ["IT_CALA", "IT_PRGP", "IT_ROSN", "MT"],
    "IT_SUD": ["GR", "IT_BRNN", "IT_CALA", "IT_CSUD", "IT_FOGN", "IT_ROSN"],
    "LT": ["BY", "LV", "PL", "RU_KGD", "SE_4"],
    "LV": ["EE", "LT", "RU"],
    "MD": ["RO", "UA", "UA_IPS"],
    "ME": ["AL", "BA", "IT_CSUD", "RS", "XK"],
    "MK": ["BG", "GR", "RS", "XK"],
    "MT": ["IT_MALTA", "IT_SICI"],
    "NL": ["BE", "DE_AT_LU", "DE_LU", "DK_1", "GB", "NO_2"],
    "NO_1": ["NO_1A", "NO_2", "NO_3", "NO_5", "SE_3"],
    "NO_1A": ["NO_1"],
    "NO_2": ["DE_LU", "DK_1", "GB", "NL", "NO_1", "NO_2A", "NO_5"],
    "NO_2A": ["NO_2"],
    "NO_3": ["NO_1", "NO_4", "NO_5", "SE_2"],
    "NO_4": ["FI", "NO_3", "SE_1", "SE_2"],
    "NO_5": ["NO_1", "NO_2", "NO_3"],
    "PL": [
        "CZ",
        "CZ_DE_SK",
        "DE_AT_LU",
        "DE_LU",
        "LT",
        "SE_4",
        "SK",
        "UA",
        "UA_DOBTPP",
        "UA_IPS",
    ],
    "PT": ["ES"],
    "RO": ["BG", "HU", "MD", "RS", "UA", "UA_BEI", "UA_IPS"],
    "RS": ["AL", "BA", "BG", "HR", "HU", "ME", "MK", "RO", "XK"],
    "RU": ["EE", "FI", "GE", "LV", "UA", "UA_IPS"],
    "RU_KGD": ["LT"],
    "SE_1": ["FI", "NO_4", "SE_2"],
    "SE_2": ["NO_3", "NO_4", "SE_1", "SE_3"],
    "SE_3": ["DK_1", "DK_1_NO_1", "FI", "NO_1", "SE_2", "SE_4"],
    "SE_4": ["DE_AT_LU", "DE_LU", "DK_1", "DK_2", "LT", "PL", "SE_3"],
    "SI": ["AT", "DE_AT_LU", "HR", "HU", "IT_NORD", "IT_NORD_SI"],
    "SK": ["CZ", "HU", "PL", "UA", "UA_BEI", "UA_IPS"],
    "TR": ["BG", "GE", "GR"],
    "UA": ["BY", "HU", "MD", "PL", "RO", "RU", "SK"],
    "UA_BEI": ["HU", "RO", "SK"],
    "UA_DOBTPP": ["PL"],
    "UA_IPS": ["BY", "HU", "MD", "PL", "RO", "RU", "SK"],
    "XK": ["AL", "ME", "MK", "RS"],
}

# https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_psrtype
# A05 and B01-B20 similar to entsoe.PSRTYPE_MAPPINGS
ALL_FEATURES = [
    "Biomass",
    "Fossil Brown coal/Lignite",
    "Fossil Coal-derived gas",
    "Fossil Gas",
    "Fossil Hard coal",
    "Fossil Oil",
    "Fossil Oil shale",
    "Fossil Peat",
    "Geothermal",
    "Hydro Pumped Storage",
    "Hydro Run-of-river and poundage",
    "Hydro Water Reservoir",
    "Load",
    "Marine",
    "Nuclear",
    "Other",
    "Other renewable",
    "Solar",
    "Waste",
    "Wind Offshore",
    "Wind Onshore",
]

# assumes the following transformations:
# all lowercase
# no spaces
# no '-', '[', ']', '/'
FEATURE_KEYWORDS = {
    "biomass": "Biomass",
    "brown": "Fossil Brown coal/Lignite",
    "derived": "Fossil Coal-derived gas",
    "fossilgas": "Fossil Gas",
    "hard": "Fossil Hard coal",
    "fossiloil": "Fossil Oil",  # fossiloil and fossiloilshale have same prefix
    "shale": "Fossil Oil shale",
    "peat": "Fossil Peat",
    "geothermal": "Geothermal",
    "storage": "Hydro Pumped Storage",
    "poundage": "Hydro Run-of-river and poundage",
    "reservoir": "Hydro Water Reservoir",
    "load": "Load",
    "marine": "Marine",
    "nuclear": "Nuclear",
    "other": "Other",  # same prefix as "other renewables"
    "otherrenewable": "Other renewable",
    "solar": "Solar",
    "waste": "Waste",
    "offshore": "Wind Offshore",
    "onshore": "Wind Onshore",
}


def name_to_feature(name: str) -> str:
    """Convert a raw name into a feature name."""
    name = name.lower()  # lowercase
    name = name.replace(" ", "")  # remove spaces
    name = name.replace("/", "")  # remove special characters
    name = name.replace("[", "")
    name = name.replace("]", "")
    name = name.replace("-", "")

    log.debug(f"Converted raw name is: {name}")

    # special case for fossil oil due to same prefix:
    if "shale" in name:
        return "Fossil Oil shale"

    if "otherrenewable" in name:
        return "Other renewable"

    for key in FEATURE_KEYWORDS:
        if key in name:
            return FEATURE_KEYWORDS[key]

    raise KeyError(f"Feature {name} not found.")


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing features to dataframe and fill values with NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe for which to add missing features.
    """
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            log.info(f"Feature {feature} is missing, add NaNs.")
            df[feature] = np.nan
    return df.sort_index(axis=1)


def get_neighbors(
    bidding_zones: List[str], relevant_only: bool
) -> List[Tuple[str, str]]:
    """
    Identify neighboring bidding zones.

    Parameters
    ----------
    bidding_zones : List[str]
        The bidding zones considered.
    relevant_only : bool
        Whether to return only neighboring bidding zones that are also part of the bidding zone list and thus of
        the graph or all neighbors on a global scale.

    Returns
    -------
    List[Tuple[str, str]]
        The neighboring pairs of bidding zones given by their string-type codes.
    """
    # Get all neighbors of each bidding zone, i.e., node considered.
    all_neighbors = {
        key: value for key, value in NEIGHBORS.items() if key in bidding_zones
    }
    # Initialize list of neighboring pairs.
    neighboring_pairs: List[Tuple[str, str]] = []

    if (
        relevant_only
    ):  # Keep only neighbors with are also nodes in the considered graph.
        neighbors: Dict[str, List[str]] = {
            key: [zone for zone in value if zone in bidding_zones]
            for key, value in all_neighbors.items()
        }

    else:  # Keep all neighbors.
        neighbors = all_neighbors

    # Convert dict of considered neighbors to set of neighboring pairs, i.e., edges.
    for zone, neighboring_zones in neighbors.items():
        for neighbor in neighboring_zones:
            # Create a tuple for each pair.
            if neighbor in bidding_zones:
                neighboring_pairs.append(tuple([zone, neighbor]))  # type:ignore

    return neighboring_pairs


class EntsoeCsvDownloader:
    """
    Obtain data from ENTSO-E transparency platform and dump to local csv files.

    Data is requested per year and bidding zone, where loads and generations are saved to separate files.
    Transmission is requested per year for each neighboring pairs of bidding zones.

    Attributes
    ----------
    client : EntsoePandasClient
        The client for downloading data from ENTSO-E RESTful API into pandas dataframes.
    bidding_zones : List[str]
        Bidding zones considered as nodes in the graph.
    intervals : List[str]
        A list of years to request data for.
    output_path : pathlib.Path | str
        The path to save data to.
    relevant_neighbors_only : bool
        Whether to consider only neighboring bidding zones that are also part of the bidding zone list and thus of
        the graph or all neighbors on a global scale.

    Methods
    -------
    dump_loads_to_csv()
        Dump load data in given zone within given time frame.
    dump_generations_to_csv()
        Dump generation data per production type in given zone within given time frame.
    dump_transmissions_to_csv()
        Dump transmission data (cross-border physical flows) from one zone to another within given time frame.
    """

    def __init__(
        self,
        client: EntsoePandasClient,
        bidding_zones: List[str],
        intervals: List[str],
        output_path: Union[pathlib.Path, str],
        relevant_neighbors_only: bool,
        resume: bool = False,
        features: List[str] = ["load", "generation", "transmission"],
    ) -> None:
        """
        Initialize an ENTSO-E CSV downloader.

        Parameters
        ----------
        client : EntsoePandasClient
            The client for downloading data from ENTSO-E RESTful API into pandas dataframes.
        bidding_zones : List[str]
            The bidding zones considered.
        intervals : List[str]
            A list of years to request data for.
        output_path : pathlib.Path | str
            The path to save data to.
        relevant_neighbors_only : bool
            Whether to consider only neighboring bidding zones that are also part of the bidding zone list and thus of
            the graph or all neighbors on a global scale.
        resume : bool
            Whether to resume from a previous download (True) or start from scratch (False).
            Default is False.
        """
        self.client = client
        self.bidding_zones = bidding_zones
        self.intervals = intervals
        self.output_path = pathlib.Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.relevant_neighbors_only = relevant_neighbors_only
        self.features = features

        self._feature_mapping: Dict[str, Tuple[str, Callable[..., int]]] = {
            "load": ("node", self.dump_loads_to_csv),
            "generation": ("node", self.dump_generations_to_csv),
            "transmission": ("edge", self.dump_transmissions_to_csv),
            "pricing": ("node", self.dump_pricing_to_csv),
        }
        for feature in self.features:
            if feature not in self._feature_mapping:
                log.warning(f"Feature {feature} is not supported.")
                self.features.remove(feature)

        self.checkpoints = {}

        if resume:
            for feature in self.features:
                feat_type, _ = self._feature_mapping[feature]
                feat_ckpt = self.output_path / f"{feature}_status.pickle"
                if feat_ckpt.is_file():
                    with open(feat_ckpt, "rb") as f:
                        self.checkpoints[feature] = pickle.load(f)
                else:
                    if feat_type == "edge":
                        self.checkpoints[feature] = np.zeros(
                            (
                                len(intervals) - 1,
                                len(
                                    get_neighbors(
                                        bidding_zones, relevant_neighbors_only
                                    )
                                ),
                            )
                        )

                    else:
                        self.checkpoints[feature] = np.zeros(
                            (len(intervals) - 1, len(bidding_zones))
                        )
        else:
            for feature in self.features:
                feat_type, _ = self._feature_mapping[feature]
                if feat_type == "edge":
                    self.checkpoints[feature] = np.zeros(
                        (
                            len(intervals) - 1,
                            len(get_neighbors(bidding_zones, relevant_neighbors_only)),
                        )
                    )
                else:
                    self.checkpoints[feature] = np.zeros(
                        (len(intervals) - 1, len(bidding_zones))
                    )
        self.dump_all_to_csv()

    def dump_loads_to_csv(self, zone: str, start: str, end: str) -> int:
        """
        Dump load data from ENTSO-E platform in given zone within given time frame to a CSV file.

        Parameters
        ----------
        zone : str
            Bidding zone to request data for.
        start : str
            The start time.
        end : str
            The end time.

        Returns
        -------
        int
            The status of the download (1 if successful, 0 if unsuccessful).
        """
        try:
            log.info(f"Dumping load for {zone} in {start} to {end}.")
            df = self.client.query_load(
                zone,
                start=pd.Timestamp(start, tz="UTC+00:00"),
                end=pd.Timestamp(end, tz="UTC+00:00"),
            )
            df.columns = [name_to_feature(col) for col in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            frequency = pd.infer_freq(df.index)
            if frequency is None:
                frequency = "h"
            log.info(f"Frequency is {frequency}.")
            df = df.asfreq(frequency)
            path_dir = self.output_path / zone / start[:4]
            path_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_dir / f"load_{zone}_{start[:4]}.csv")
            return 1
        except Exception as e:
            if isinstance(e, NoMatchingDataError):
                log.info("No data available. Setting download status to 1.")
                return 1
            else:
                log.info(e)
                return 0

    def dump_generations_to_csv(self, zone: str, start: str, end: str) -> int:
        """
        Dump generation data per production type from ENTSO-E platform in given zone within given time frame.

        Parameters
        ----------
        zone : str
            Bidding zone to request data for.
        start : str
            The start time.
        end : str
            The end time.

        Returns
        -------
        int
            The status of the download (1 if successful, 0 if unsuccessful).
        """
        try:
            log.info(f"Dumping generation for {zone} in {start} to {end}.")
            df = self.client.query_generation(
                zone,
                start=pd.Timestamp(start, tz="UTC+00:00"),
                end=pd.Timestamp(end, tz="UTC+00:00"),
                nett=True,
            )
            df.columns = [" ".join(col) for col in df.columns]
            df.columns = [name_to_feature(col) for col in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            frequency = pd.infer_freq(df.index)
            if frequency is None:
                frequency = "h"
            log.info(f"Frequency is {frequency}.")
            df = df.asfreq(frequency)
            df = add_missing_features(df)
            df.drop(columns="Load", inplace=True)

            path_dir = self.output_path / zone / start[:4]
            path_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_dir / f"generation_{zone}_{start[:4]}.csv")
            return 1
        except Exception as e:
            if isinstance(e, NoMatchingDataError):
                log.info("No data available. Setting download status to 1.")
                return 1
            else:
                log.info(e)
                return 0

    def dump_transmissions_to_csv(
        self, zone_from: str, zone_to: str, start: str, end: str
    ) -> int:
        """
        Dump transmission data (cross-border physical flows) from one zone to another within given time frame.

        Parameters
        ----------
        zone_from : str
            Source bidding zone.
        zone_to : str
            Target bidding zone.
        start : str
            The start time.
        end : str
            The end time.

        Returns
        -------
        int
            The status of the download (1 if successful, 0 if unsuccessful).
        """
        try:
            log.info(
                f"Dumping transmission from {zone_from} to {zone_to} in {start} to {end}."
            )
            ts = self.client.query_crossborder_flows(
                zone_from,
                zone_to,
                start=pd.Timestamp(start, tz="UTC+00:00"),
                end=pd.Timestamp(end, tz="UTC+00:00"),
            )
            ts.name = f"{zone_from}>{zone_to}".lower()
            ts.index = pd.to_datetime(ts.index, utc=True)
            frequency = pd.infer_freq(ts.index)
            if frequency is None:
                frequency = "h"
            log.info(f"Frequency is {frequency}.")
            ts = ts.asfreq(frequency)

            path_dir = self.output_path / zone_from / start[:4]
            path_dir.mkdir(parents=True, exist_ok=True)
            ts.to_csv(path_dir / f"transmission_{zone_from}->{zone_to}_{start[:4]}.csv")
            return 1
        except Exception as e:
            if isinstance(e, NoMatchingDataError):
                log.info("No data available. Setting download status to 1.")
                return 1
            else:
                log.info(e)
                return 0

    def dump_pricing_to_csv(self, zone: str, start: str, end: str) -> int:
        """
        Dump raw pricing data from ENTSO-E platform in given zone within given time frame.

        Parameters
        ----------
        zone : str
            Bidding zone to request data for.
        start : str
            The start time.
        end : str
            The end time.
        """
        try:
            log.info(f"Dumping raw pricing for {zone} in {start} to {end}.")
            df = self.client.query_day_ahead_prices(
                zone,
                start=pd.Timestamp(start, tz="UTC+00:00"),
                end=pd.Timestamp(end, tz="UTC+00:00"),
            )
            df.index = pd.to_datetime(df.index, utc=True)
            frequency = pd.infer_freq(df.index)
            if frequency is None:
                frequency = "h"
            df = df.asfreq(frequency)
            df = df.to_frame()  # convert from pd.Series to pd.Dataframe
            log.info(f"Frequency is {frequency}.")

            path_dir = self.output_path / zone / start[:4]
            path_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(path_dir / f"pricing_{zone}_{start[:4]}.csv")
            return 1
        except Exception as e:
            if isinstance(e, NoMatchingDataError):
                log.info("No data available. Setting download status to 1.")
                return 1
            else:
                log.info(e)
                return 0

    def dump_all_to_csv(self) -> None:
        """Dump all data for all requested zones and years."""
        neighbors = get_neighbors(self.bidding_zones, self.relevant_neighbors_only)
        log.info(f"Neighboring pairs are: {neighbors}")
        for idx, _ in enumerate(self.intervals[:-1]):
            for ibz, zone in enumerate(self.bidding_zones):
                for feature in self.features:
                    feat_type, feat_func = self._feature_mapping[feature]
                    if feat_type == "node":
                        if self.checkpoints[feature][idx, ibz] == 0:
                            self.checkpoints[feature][idx, ibz] = feat_func(
                                zone, self.intervals[idx], self.intervals[idx + 1]
                            )
                            with open(
                                self.output_path / f"{feature}_status.pickle", "wb"
                            ) as f:
                                pickle.dump(self.checkpoints[feature], f)
                        else:
                            log.info(
                                f"{feature.capitalize()} for {zone} in {self.intervals[idx]} to {self.intervals[idx + 1]} already downloaded."
                            )
            for inp, neighboring_pair in enumerate(neighbors):
                for feature in self.features:
                    feat_type, feat_func = self._feature_mapping[feature]
                    if feat_type == "edge":
                        if self.checkpoints[feature][idx, inp] == 0:
                            self.checkpoints[feature][idx, inp] = feat_func(
                                neighboring_pair[0],
                                neighboring_pair[1],
                                self.intervals[idx],
                                self.intervals[idx + 1],
                            )
                            with open(
                                self.output_path / f"{feature}_status.pickle", "wb"
                            ) as f:
                                pickle.dump(self.checkpoints[feature], f)
                        else:
                            log.info(
                                f"{feature.capitalize()} for {neighboring_pair[0]} to {neighboring_pair[1]} in {self.intervals[idx]} to {self.intervals[idx + 1]} already downloaded."
                            )


if __name__ == "__main__":
    # -------- Config --------
    # Set output path to save downloaded CSV files to.
    # You can adjust the corresponding environmental variable in the download.env.sh file which you need to source
    # before running the script.

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="entsoe/")
    parser.add_argument("--bidding-zones", type=str, default="FR")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True)

    args = parser.parse_args()

    path_suffix = args.output_path
    output_path = pathlib.Path(os.environ.get("DATA_DIR", ".")) / path_suffix
    output_path.mkdir(parents=True, exist_ok=True)

    # Download data for the following years:
    intervals = [
        "20150101",  # NOTE: Transparency platform was officially launched on Jan, 5 2015.
        "20160101",
        "20170101",
        "20180101",
        "20190101",
        "20200101",
        "20210101",
        "20220101",
        "20230101",
        "20240101",
        "20240601",
    ]

    # Bidding zones considered as nodes in the graph.
    # Change to your assigned subset.
    if args.bidding_zones is not None:
        your_bidding_zones = args.bidding_zones.split(",")
    else:
        your_bidding_zones = ALL_BIDDING_ZONES.copy()

    # Features to download.
    # Change to your assigned subset.
    if args.features is not None:
        your_features = args.features.split(",")
    else:
        your_features = ["load", "generation", "pricing"]

    # -------- Download --------
    # Set up ENTSO-E pandas client using your personal secret API token.
    # Make your token available as an environment variable by putting it into the download.env.sh file and sourcing
    # this file in the terminal where you run this Python script.
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_TOKEN"])

    EntsoeCsvDownloader(
        client=client,
        bidding_zones=your_bidding_zones,
        intervals=intervals,
        output_path=output_path,
        resume=args.resume,  # Change this if you need to restart your download bc of exceptions, timeout, etc.
        relevant_neighbors_only=False,  # Do not change this.
        features=your_features,
        # The option ensures that you download the transmissions for all neighbors of your bidding zones and not only
        # those neighbors that are also part of your assigned neighbors (which might be none).
    )
