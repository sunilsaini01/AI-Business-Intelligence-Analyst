import pytest

from app.tools.schema_tools import ALLOWED_SCHEMAS, format_schema_for_prompt, get_analytics_schema


@pytest.mark.asyncio
async def test_schema_covers_both_allowed_schemas():
    schema = await get_analytics_schema()
    schemas_seen = {key.split(".")[0] for key in schema}
    assert schemas_seen == set(ALLOWED_SCHEMAS)


@pytest.mark.asyncio
async def test_analytics_and_olist_customers_have_different_columns():
    schema = await get_analytics_schema()
    analytics_cols = set(schema["analytics.customers"])
    olist_cols = set(schema["olist.customers"])
    assert "segment" in analytics_cols
    assert "segment" not in olist_cols
    assert "customer_unique_id" in olist_cols
    assert "customer_unique_id" not in analytics_cols


@pytest.mark.asyncio
async def test_format_schema_for_prompt_filters_by_schema():
    schema = await get_analytics_schema()
    text = format_schema_for_prompt(schema, only_schema="olist")
    assert "olist.customers" in text
    assert "analytics.customers" not in text
