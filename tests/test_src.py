from src import config, markowitz, __version__
import unittest
import pandas as pd
import numpy as np
import logging
import scipy

logging.basicConfig(level=logging.INFO, filename="result.log")


class test_src(unittest.TestCase):
    def setUp(self):
        company, data = config.prepare_data()
        self.assertIsInstance(company, dict, "failed to parse data")
        logging.info(f"companys: {tuple(company.values())}")
        self.comp = company
        self.data = data
        markow = markowitz.Markowitz(data)
        proportion = markow.solveMinVar(config.expected_return)
        self.assertIsInstance(proportion, pd.DataFrame, "failed to solve min var")
        logging.info(f"proportion: {tuple(proportion.values)}")
        self.proportion = proportion

    def test_src(self):
        self.assertEqual(__version__, "0.1.0", msg="python environment is not correct")

    def test_historical(self):
        from src import historical

        var_10 = historical.var(self.data, self.proportion, days=10)
        var_60 = historical.var(self.data, self.proportion, days=60)
        self.assertAlmostEqual(var_10 * np.sqrt(6), var_60)
        logging.info(f"VaR {historical.__name__} in 10 days {var_10}")
        logging.info(f"VaR {historical.__name__} in 60 days {var_60}")

    def test_cov(self):
        from src import var_cov

        var_10 = var_cov.var(self.data, self.proportion, days=10)
        var_60 = var_cov.var(self.data, self.proportion, days=60)
        self.assertAlmostEqual(var_10 * np.sqrt(6), var_60)
        logging.info(f"VaR {var_cov.__name__} in 10 days {var_10}")
        logging.info(f"VaR {var_cov.__name__} in 60 days {var_60}")

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
        logging.info(f"VaR {monte_carlo.__name__} in 10 days {var_10}")
        logging.info(f"VaR {monte_carlo.__name__} in 60 days {var_60}")
        self.assertAlmostEqual(
            var_10 * np.sqrt(6) / config.initial_investment,
            var_60 / config.initial_investment,
            delta=0.1,
        )


if __name__ == "__main__":
    unittest.main()
