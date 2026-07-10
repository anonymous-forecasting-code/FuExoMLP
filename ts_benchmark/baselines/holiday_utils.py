from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd


HOLIDAY_FILTER_NONE = {"", "none", "off", "false", "0", None}
HOLIDAY_ROW_FILTER_STAGES = {"row", "rows", "raw", "source", "data", "dataset"}


def is_holiday_filter_enabled(value) -> bool:
    return str(value).strip().lower() not in HOLIDAY_FILTER_NONE


def normalize_holiday_filter_stage(value) -> str:
    return str(value or "sample").strip().lower()


def is_holiday_row_filter_stage(value) -> bool:
    return normalize_holiday_filter_stage(value) in HOLIDAY_ROW_FILTER_STAGES


def infer_holiday_calendar(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = Path(str(name)).stem.lower()
    aliases = {
        "np": "nordic",
        "pjm": "us",
        "be": "belgium",
        "fr": "france",
        "de": "germany",
        "energy": "chile",
        "colbun": "chile",
        "rapel": "chile",
        "sdwpfm1": "china",
        "sdwpfm2": "china",
        "sdwpfh1": "china",
        "sdwpfh2": "china",
    }
    return aliases.get(key, key)


def normalize_holiday_calendar(calendar: Optional[str]) -> Optional[str]:
    if calendar is None:
        return None
    key = str(calendar).strip().lower()
    if key == "auto":
        return key
    aliases = {
        "np": "nordic",
        "nordpool": "nordic",
        "nord_pool": "nordic",
        "nordic-common": "nordic",
        "pjm": "us",
        "usa": "us",
        "united_states": "us",
        "be": "belgium",
        "bel": "belgium",
        "fr": "france",
        "fra": "france",
        "de": "germany",
        "deu": "germany",
        "ger": "germany",
        "cl": "chile",
        "chl": "chile",
        "cn": "china",
        "chn": "china",
    }
    return aliases.get(key, key)


def resolve_holiday_calendar(calendar: Optional[str], dataset_name: Optional[str] = None):
    calendar = normalize_holiday_calendar(calendar)
    if calendar in (None, "auto"):
        calendar = infer_holiday_calendar(dataset_name)
    return normalize_holiday_calendar(calendar)


def _easter(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed_fixed(year: int, month: int, day: int) -> date:
    fixed = date(year, month, day)
    if fixed.weekday() == 5:
        return fixed - timedelta(days=1)
    if fixed.weekday() == 6:
        return fixed + timedelta(days=1)
    return fixed


def _date_range(start: date, end: date) -> Set[date]:
    days = set()
    current = start
    while current <= end:
        days.add(current)
        current += timedelta(days=1)
    return days


def _nordic_common_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        easter = _easter(year)
        days |= {
            date(year, 1, 1),
            easter - timedelta(days=2),
            easter + timedelta(days=1),
            date(year, 5, 1),
            easter + timedelta(days=39),
            date(year, 12, 25),
            date(year, 12, 26),
        }
    return days


def _us_federal_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        for month, day in ((1, 1), (7, 4), (11, 11), (12, 25)):
            days.add(date(year, month, day))
            days.add(_observed_fixed(year, month, day))
        days |= {
            _nth_weekday(year, 1, 0, 3),
            _nth_weekday(year, 2, 0, 3),
            _last_weekday(year, 5, 0),
            _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 10, 0, 2),
            _nth_weekday(year, 11, 3, 4),
        }
    return days


def _belgium_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        easter = _easter(year)
        days |= {
            date(year, 1, 1),
            easter + timedelta(days=1),
            date(year, 5, 1),
            easter + timedelta(days=39),
            easter + timedelta(days=50),
            date(year, 7, 21),
            date(year, 8, 15),
            date(year, 11, 1),
            date(year, 11, 11),
            date(year, 12, 25),
        }
    return days


def _france_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        easter = _easter(year)
        days |= {
            date(year, 1, 1),
            easter + timedelta(days=1),
            date(year, 5, 1),
            date(year, 5, 8),
            easter + timedelta(days=39),
            easter + timedelta(days=50),
            date(year, 7, 14),
            date(year, 8, 15),
            date(year, 11, 1),
            date(year, 11, 11),
            date(year, 12, 25),
        }
    return days


def _germany_national_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        easter = _easter(year)
        days |= {
            date(year, 1, 1),
            easter - timedelta(days=2),
            easter + timedelta(days=1),
            date(year, 5, 1),
            easter + timedelta(days=39),
            easter + timedelta(days=50),
            date(year, 10, 3),
            date(year, 12, 25),
            date(year, 12, 26),
        }
    return days


def _chile_holidays(years: Iterable[int]) -> Set[date]:
    days = set()
    for year in years:
        easter = _easter(year)
        days |= {
            date(year, 1, 1),
            easter - timedelta(days=2),
            easter - timedelta(days=1),
            date(year, 5, 1),
            date(year, 5, 21),
            date(year, 6, 29),
            date(year, 7, 16),
            date(year, 8, 15),
            date(year, 9, 18),
            date(year, 9, 19),
            date(year, 10, 12),
            date(year, 10, 31),
            date(year, 11, 1),
            date(year, 12, 8),
            date(year, 12, 25),
        }
    return days


def _china_holiday_leave_days(years: Iterable[int]) -> Set[date]:
    years = set(years)
    ranges = []
    if 2020 in years:
        ranges += [
            (date(2020, 5, 1), date(2020, 5, 5)),
            (date(2020, 6, 25), date(2020, 6, 27)),
            (date(2020, 10, 1), date(2020, 10, 8)),
        ]
    if 2021 in years:
        ranges += [
            (date(2021, 1, 1), date(2021, 1, 3)),
            (date(2021, 2, 11), date(2021, 2, 17)),
            (date(2021, 4, 3), date(2021, 4, 5)),
            (date(2021, 5, 1), date(2021, 5, 5)),
            (date(2021, 6, 12), date(2021, 6, 14)),
            (date(2021, 9, 19), date(2021, 9, 21)),
            (date(2021, 10, 1), date(2021, 10, 7)),
        ]
    if 2022 in years:
        ranges.append((date(2022, 1, 1), date(2022, 1, 3)))

    days = set()
    for start, end in ranges:
        days |= _date_range(start, end)
    return days


CALENDAR_BUILDERS = {
    "nordic": _nordic_common_holidays,
    "us": _us_federal_holidays,
    "belgium": _belgium_holidays,
    "france": _france_holidays,
    "germany": _germany_national_holidays,
    "chile": _chile_holidays,
    "china": _china_holiday_leave_days,
}


def holiday_dates_for_index(index: pd.Index, calendar: Optional[str], dataset_name: Optional[str] = None) -> Set[date]:
    calendar = resolve_holiday_calendar(calendar, dataset_name)
    if calendar not in CALENDAR_BUILDERS:
        raise ValueError(f"Unknown holiday calendar: {calendar!r}")

    dt_index = pd.DatetimeIndex(index)
    years = range(dt_index.min().year - 1, dt_index.max().year + 2)
    all_days = CALENDAR_BUILDERS[calendar](years)
    min_day = dt_index.min().date()
    max_day = dt_index.max().date()
    return {day for day in all_days if min_day <= day <= max_day}


def holiday_mask_for_index(index: pd.Index, calendar: Optional[str], dataset_name: Optional[str] = None) -> np.ndarray:
    days = holiday_dates_for_index(index, calendar, dataset_name)
    if not days:
        return np.zeros(len(index), dtype=bool)
    dates = pd.DatetimeIndex(index).normalize().date
    return np.array([day in days for day in dates], dtype=bool)


def non_holiday_frame(data: pd.DataFrame, calendar: Optional[str], dataset_name: Optional[str] = None) -> pd.DataFrame:
    mask = holiday_mask_for_index(data.index, calendar, dataset_name)
    if not mask.any() or mask.all():
        return data
    return data.iloc[~mask]
