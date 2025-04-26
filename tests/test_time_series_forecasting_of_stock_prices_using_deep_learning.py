#!/usr/bin/env python

"""Tests for `time_series_forecasting_of_stock_prices_using_deep_learning` package."""

import pytest


from time_series_forecasting_of_stock_prices_using_deep_learning import time_series_forecasting_of_stock_prices_using_deep_learning


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
