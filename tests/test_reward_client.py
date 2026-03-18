from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.strategy.reward_client import RewardClient, MarketRewardInfo


def _mock_response(json_data):
    """Create a mock httpx response with sync .json() and .raise_for_status()."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def client(settings: Settings) -> RewardClient:
    c = RewardClient(settings)
    c.client = MagicMock()
    return c


class TestRewardClient:
    def test_parse_reward_market_full(self, client: RewardClient):
        raw = {
            "condition_id": "0xabc",
            "question": "Will X happen?",
            "rewards_daily_rate": 150.0,
            "max_spread": 200,
            "min_size": 100,
            "num_lps": 5,
            "competition_score": 2.5,
            "volume_24h": 50000,
            "active": True,
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_abc"},
                {"outcome": "No", "token_id": "no_abc"},
            ],
        }

        result = client._parse_reward_market(raw)
        assert result is not None
        assert result.market_id == "0xabc"
        assert result.rate_per_day == 150.0
        assert result.max_spread_bps == 200
        assert result.min_shares == 100
        assert result.num_lps == 5
        assert result.competition_score == 2.5
        assert result.yes_token_id == "yes_abc"
        assert result.no_token_id == "no_abc"
        assert result.volume_24h == 50000

    def test_parse_reward_market_minimal(self, client: RewardClient):
        raw = {"id": "0xmin", "tokens": []}
        result = client._parse_reward_market(raw)
        assert result is not None
        assert result.market_id == "0xmin"
        assert result.rate_per_day == 0.0
        assert result.yes_token_id == ""

    def test_parse_reward_market_no_id(self, client: RewardClient):
        raw = {"question": "no id market", "tokens": []}
        result = client._parse_reward_market(raw)
        assert result is None

    def test_parse_reward_market_fallback_fields(self, client: RewardClient):
        """rate_per_day field should work as fallback."""
        raw = {
            "market_id": "0xfallback",
            "rate_per_day": 75.0,
            "tokens": [],
        }
        result = client._parse_reward_market(raw)
        assert result is not None
        assert result.rate_per_day == 75.0

    @pytest.mark.asyncio
    async def test_fetch_reward_markets_list_response(self, client: RewardClient):
        """API returns a list directly."""
        mock_resp = _mock_response([
            {"id": "m1", "rewards_daily_rate": 10.0, "tokens": []},
            {"id": "m2", "rewards_daily_rate": 0.5, "tokens": []},  # Below threshold
        ])
        client.client.get = AsyncMock(return_value=mock_resp)

        results = await client.fetch_reward_markets()
        # m2 has rate 0.5 < min_reward_rate_daily=1.0, so filtered
        assert len(results) == 1
        assert results[0].market_id == "m1"

    @pytest.mark.asyncio
    async def test_fetch_reward_markets_dict_response(self, client: RewardClient):
        """API returns {"markets": [...]}."""
        mock_resp = _mock_response({
            "markets": [
                {"id": "m1", "rewards_daily_rate": 5.0, "tokens": []},
            ]
        })
        client.client.get = AsyncMock(return_value=mock_resp)

        results = await client.fetch_reward_markets()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_fetch_reward_markets_error(self, client: RewardClient):
        """API error returns empty list."""
        client.client.get = AsyncMock(side_effect=Exception("connection failed"))
        results = await client.fetch_reward_markets()
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_market_competition(self, client: RewardClient):
        mock_resp = _mock_response({"top_lps": [], "score": 5.0})
        client.client.get = AsyncMock(return_value=mock_resp)

        result = await client.fetch_market_competition("0xabc")
        assert "score" in result

    @pytest.mark.asyncio
    async def test_fetch_market_competition_error(self, client: RewardClient):
        client.client.get = AsyncMock(side_effect=Exception("fail"))
        result = await client.fetch_market_competition("0xabc")
        assert result == {}
