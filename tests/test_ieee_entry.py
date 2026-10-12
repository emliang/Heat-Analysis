"""Entry-point checks that do not run the IEEE OPF solver."""

import contextlib
import io
import os
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import TDOPF_ieee


class IEEEEntryTests(unittest.TestCase):
    def run_entry(self, smoke):
        with tempfile.TemporaryDirectory() as output_dir:
            os.makedirs(os.path.join(output_dir, 'IEEE30'))
            with patch.object(TDOPF_ieee, 'MODELS', output_dir), patch.object(
                TDOPF_ieee, 'TDACOPF', side_effect=lambda _, df, *args: df
            ) as run, contextlib.redirect_stdout(io.StringIO()):
                TDOPF_ieee.main(smoke=smoke)
            return [call.args for call in run.call_args_list]

    def test_smoke_runs_two_modes_without_overwriting_full_sweep(self):
        calls = self.run_entry(smoke=True)
        self.assertEqual([args[-1] for args in calls],
                         ['base', 'td_derate_iter_2'])
        self.assertTrue(all(args[0] == 'heatflow_smoke' for args in calls))

    def test_default_keeps_full_experiment_matrix(self):
        calls = self.run_entry(smoke=False)
        self.assertEqual(len(calls), 108)
        self.assertEqual(set(args[0] for args in calls), {'heatflow_analysis'})
        self.assertEqual(set(Counter(args[-1] for args in calls).values()), {12})


if __name__ == '__main__':
    unittest.main()
