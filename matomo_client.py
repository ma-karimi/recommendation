from __future__ import annotations
import datetime as dt
from typing import Literal, Sequence

import polars as pl
import requests

from .settings import load_config


class MatomoClient:
    def __init__(self, base_url: str, site_id: str, token_auth: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.site_id = site_id
        self.token_auth = token_auth
        self.verify_ssl = verify_ssl

    def _get(self, method: str, params: dict) -> dict:
        url = f"{self.base_url}/index.php"
        q = {
            "module": "API",
            "method": method,
            "idSite": self.site_id,
            "format": "json",
            "token_auth": self.token_auth,
        }
        q.update(params)
        headers = {
            "User-Agent": "rochi-reco/1.0",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = None
        try:
            # Some Matomo configurations require token_auth to be sent via POST.
            resp = requests.post(url, data=q, timeout=60, headers=headers, verify=self.verify_ssl)
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = resp.status_code if resp is not None else (e.response.status_code if e.response is not None else "?")
            body = (resp.text if resp is not None else (e.response.text if e.response is not None else ""))[:500]
            msg = f"Matomo API error {status} for method={method}: {body}"
            raise requests.HTTPError(msg) from e
        return resp.json()

    def fetch_events(self, period: Literal["day", "week", "month", "range"], date: str, segment: str | None = None) -> pl.DataFrame:
        """
        Fetch events (log events) aggregated from Matomo Events API.
        For raw logs, Matomo Log Analytics or custom archiving is needed; here we use events summary.
        """
        params = {"period": period, "date": date}
        if segment:
            params["segment"] = segment
        data = self._get("Events.getAction", params)
        df = pl.DataFrame(data) if isinstance(data, list) else pl.DataFrame([data])
        return df

    def fetch_pageviews(self, period: Literal["day", "week", "month", "range"], date: str, segment: str | None = None) -> pl.DataFrame:
        params = {"period": period, "date": date}
        if segment:
            params["segment"] = segment
        data = self._get("Actions.getPageUrls", params)
        df = pl.DataFrame(data) if isinstance(data, list) else pl.DataFrame([data])
        return df

    def fetch_goals(self, period: Literal["day", "week", "month", "range"], date: str, segment: str | None = None) -> pl.DataFrame:
        params = {"period": period, "date": date}
        if segment:
            params["segment"] = segment
        data = self._get("Goals.get", params)
        df = pl.DataFrame(data) if isinstance(data, list) else pl.DataFrame([data])
        return df


def load_matomo_client() -> MatomoClient:
    cfg = load_config()
    return MatomoClient(cfg.matomo.base_url, cfg.matomo.site_id, cfg.matomo.token_auth, verify_ssl=cfg.matomo.verify_ssl)


def build_events_frame(client: MatomoClient, start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """Build a consolidated events frame for a date range using Polars."""
    # Matomo expects dates in YYYY-MM-DD, and period=range with date=f"{start},{end}"
    period = "range"
    date = f"{start_date:%Y-%m-%d},{end_date:%Y-%m-%d}"
    events_df = client.fetch_events(period=period, date=date)
    pageviews_df = client.fetch_pageviews(period=period, date=date)
    goals_df = client.fetch_goals(period=period, date=date)

    # basic normalization: ensure common columns exist
    def with_date(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
        return df.with_columns(pl.lit(date).alias(col))

    events_df = with_date(events_df)
    pageviews_df = with_date(pageviews_df)
    goals_df = with_date(goals_df)

    events_sel = [c for c in events_df.columns]
    page_sel = [c for c in pageviews_df.columns]
    goals_sel = [c for c in goals_df.columns]

    out = {
        "events": events_df.select(events_sel),
        "pageviews": pageviews_df.select(page_sel),
        "goals": goals_df.select(goals_sel),
    }
    return pl.DataFrame({k: [v] for k, v in out.items()})