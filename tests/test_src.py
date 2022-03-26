import logging
import unittest

import numpy as np
import pandas as pd

from src import __version__, config, markowitz

logging.basicConfig(level=logging.INFO, filename="result.log", format="%(message)s")


class TestSrc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.company, cls.data = config.prepare_data()
        markow = markowitz.Markowitz(cls.data)
        proportion = markow.solveMinVar(config.expected_return)
        cls.proportion = proportion

    def setUp(self):
        self.comp = TestSrc.company
        self.data = TestSrc.data
        self.proportion = TestSrc.proportion

    def test_data(self):
        self.assertIsInstance(self.comp, dict, "failed to parse data")
        self.assertIsInstance(self.proportion, pd.DataFrame, "failed to solve min var")
        proportion = self.proportion.to_dict()[0]
        for comp in self.comp:
            logging.info(
                "company %s with proportion %s %%",
                self.comp[comp][0],
                str(round(proportion[comp] * 100, 2)),
            )

    def test_src(self):
        self.assertEqual(__version__, "0.1.0", msg="python environment is not correct")

    def test_historical(self):
        from src import historical

        var_10 = historical.var(self.data, self.proportion, days=10)
        var_60 = historical.var(self.data, self.proportion, days=60)
        self.assertAlmostEqual(var_10 * np.sqrt(6), var_60)
        logging.info("Historical VaR in 10 days %s", str(var_10))
        logging.info("Historical VaR in 60 days %s", str(var_60))

    def test_cov(self):
        from src import var_cov

        var_10 = var_cov.var(self.data, self.proportion, days=10)
        var_60 = var_cov.var(self.data, self.proportion, days=60)
        self.assertAlmostEqual(var_10 * np.sqrt(6), var_60)
        logging.info("variance-covariance VaR in 10 days %s", str(var_10))
        logging.info("variance-covariance VaR in 60 days %s", str(var_60))

    def test_monte(self):
        from src import monte_carlo

        var_10 = monte_carlo.var(
            self.data,
            self.proportion,
            n1=1000,
            n2=100,
            price={i: self.comp[i][1] for i in self.comp},
        )
        var_60 = monte_carlo.var(
            self.data,
            self.proportion,
            days=60,
            n1=1000,
            n2=100,
            price={i: self.comp[i][1] for i in self.comp},
        )
        logging.info("monte-carlo VaR in 10 days %s", str(var_10))
        logging.info("monte-carlo VaR in 60 days %s", str(var_60))
        self.assertAlmostEqual(
            var_10 * np.sqrt(6) / config.initial_investment,
            var_60 / config.initial_investment,
            delta=0.2,
        )


if __name__ == "__main__":
    unittest.main()
