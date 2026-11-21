"""Backend-only checks for explicit capped-height warnings and strict legacy jobs."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from height_warning_policy import check_height_outliers, height_outlier_counts


class CourseHeightPolicyTests(unittest.TestCase):
    def report(self,**updates):
        result=dict(height_outlier_policy='clamp_warn',support_normal_over_limit_points=2,
            support_normal_over_limit_clamped_points=2)
        result.update(updates)
        return result

    def test_caps_remain_explicit_warnings(self):
        with redirect_stdout(StringIO()) as output:
            accepted=check_height_outliers(dict(support_validation=self.report()))
        self.assertEqual(accepted,2)
        self.assertIn('physical gaps remain',output.getvalue())

    def test_legacy_outliers_still_block(self):
        with self.assertRaisesRegex(ValueError,'exceed'):
            check_height_outliers(dict(support_validation=dict(support_normal_over_limit_points=2)))

    def test_incomplete_or_strict_caps_still_block(self):
        for overrides in (dict(support_normal_over_limit_clamped_points=1),dict(height_outlier_policy='require_refinement')):
            with self.assertRaisesRegex(ValueError,'exceed'):
                check_height_outliers(dict(support_validation=self.report(**overrides)))

    def test_malformed_counts_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'Invalid'):
            height_outlier_counts(self.report(support_normal_over_limit_clamped_points=3))

    def test_only_the_selected_main_range_is_checked(self):
        metadata=dict(layers=[dict(layer_main=0,support_validation=dict(support_normal_over_limit_points=2)),
                              dict(layer_main=1,support_validation=self.report())])
        self.assertEqual(check_height_outliers(metadata,start_main=1,end_main=1,warn=False),2)
        self.assertEqual(check_height_outliers(metadata,start_main=2,end_main=3,warn=False),0)
        with self.assertRaises(ValueError):check_height_outliers(metadata,start_main=0,end_main=1)

    def test_summary_only_jobs_are_supported(self):
        self.assertEqual(check_height_outliers(dict(layers=[dict(layer_main=1)],support_validation=self.report()),warn=False),2)


if __name__=='__main__':unittest.main()
