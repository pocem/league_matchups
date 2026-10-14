"""Minimal async Riot API client with rate limiting driven by Riot's own headers."""
from __future__ import annotations

import asyncio
import bisect
import logging
import time

import aiohttp

log = logging.getLogger(__name__)

# Development and personal keys: 20 requests / 1 s and 100 requests / 2 min.
# The real limits are read from X-App-Rate-Limit after the first response.
DEFAULT_APP_LIMITS = [(20, 1.0), (100, 120.0)]

PLATFORM_ROUTING = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}


class AuthError(RuntimeError):
    """The key was rejected: expired dev key, wrong key, or endpoint not allowed."""


class RiotError(RuntimeError):
    pass


def parse_limits(header: str | None) -> list[tuple[int, float]] | None:
    """'20:1,100:120' -> [(20, 1.0), (100, 120.0)]"""
    if not header:
        return None
    limits = []
    for part in header.split(","):
        count, secs = part.split(":")
        limits.append((int(count), float(secs)))
    return limits or None


class RateLimiter:
    """Sliding-window limiter that enforces several (count, seconds) windows at once.

    Waiters queue on a lock, so slots are handed out first come, first served.
    """

    def __init__(self, limits, slack: float = 0.25):
        self.limits = list(limits)
        self.slack = slack
        self._stamps: list[float] = []
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    def set_limits(self, limits):
        self.limits = list(limits)

    def block(self, seconds: float):
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)

    async def acquire(self):
        async with self._lock:
            while True:
                now = time.monotonic()
                wait = self._blocked_until - now
                if self.limits:
                    longest = max(secs for _, secs in self.limits)
                    del self._stamps[:bisect.bisect_right(self._stamps, now - longest)]
                n = len(self._stamps)
                for count, secs in self.limits:
                    in_window = n - bisect.bisect_right(self._stamps, now - secs)
                    if in_window >= count:
                        # wait until the oldest request that keeps us at the cap ages out
                        wait = max(wait, self._stamps[n - count] + secs + self.slack - now)
                if wait <= 0:
                    self._stamps.append(now)
                    return
                await asyncio.sleep(wait)


class RiotClient:
    """GET-only client. Each host (platform like 'kr', routing like 'asia') has its own limits."""

    def __init__(self, api_key: str, app_limits=DEFAULT_APP_LIMITS, timeout: float = 30,
                 base_url: str = "https://{host}.api.riotgames.com"):
        self.api_key = api_key
        self.app_limits = list(app_limits)
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.requests = 0
        self._app: dict[str, RateLimiter] = {}
        self._method: dict[tuple[str, str], RateLimiter] = {}
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers={"X-Riot-Token": self.api_key},
                                              timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    def _app_limiter(self, host: str) -> RateLimiter:
        if host not in self._app:
            self._app[host] = RateLimiter(self.app_limits)
        return self._app[host]

    def _update_limits(self, host: str, method: str, headers):
        app = parse_limits(headers.get("X-App-Rate-Limit"))
        if app and app != self._app[host].limits:
            log.info("%s: app rate limits %s", host, app)
            self._app[host].set_limits(app)
        meth = parse_limits(headers.get("X-Method-Rate-Limit"))
        if meth:
            limiter = self._method.get((host, method))
            if limiter is None:
                self._method[(host, method)] = RateLimiter(meth)
            elif limiter.limits != meth:
                limiter.set_limits(meth)

    async def get(self, host: str, path: str, params=None, method: str | None = None,
                  max_failures: int = 6):
        """Return parsed JSON, or None on 404. Retries 429s and 5xx up to max_failures times,
        and network errors until the connection comes back."""
        url = self.base_url.format(host=host) + path
        method = method or path
        app = self._app_limiter(host)
        failures = network_failures = 0
        while True:
            await app.acquire()
            meth = self._method.get((host, method))
            if meth:
                await meth.acquire()
            try:
                async with self._session.get(url, params=params) as r:
                    self.requests += 1
                    self._update_limits(host, method, r.headers)
                    if r.status == 200:
                        return await r.json()
                    if r.status == 404:
                        return None
                    if r.status in (401, 403):
                        raise AuthError(f"{r.status} from {host}{path}: key expired, invalid, "
                                        "or not allowed for this endpoint")
                    if r.status == 429:
                        wait = float(r.headers.get("Retry-After", 5))
                        kind = r.headers.get("X-Rate-Limit-Type", "service")
                        log.warning("429 (%s limit) on %s, backing off %.0fs", kind, host, wait)
                        if kind == "application":
                            app.block(wait)
                        elif kind == "method" and meth:
                            meth.block(wait)
                        else:
                            await asyncio.sleep(wait)
                        continue
                    if r.status < 500:
                        raise RiotError(f"{r.status} from {host}{path}: {await r.text()}")
                    log.warning("%s from %s%s", r.status, host, path)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # connection or DNS trouble hits every request alike, so wait it out instead of giving up
                log.warning("network error on %s%s: %r", host, path, e)
                network_failures += 1
                await asyncio.sleep(min(60, 2 ** network_failures))
                continue
            failures += 1
            if failures >= max_failures:
                raise RiotError(f"gave up on {host}{path} after {failures} failures")
            await asyncio.sleep(min(60, 2 ** failures))
