import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p
import utils as u
from calculate_english_text import pair_columns, calculate_radeval
from calculate_russian_text import ManualGREENQwen
from model_loading import require_complete_generation
from translate_reports import _translate_one_translator_columns


class MetricTests(unittest.TestCase):
    def test_translation_loads_once_for_candidate_and_reference(self):
        df = pd.DataFrame({'candidate_report':['c1','c2'], 'reference_report':['r1','r2']})
        calls = []
        def translate(texts, *args):
            calls.append(list(texts))
            return ['en:'+t for t in texts]
        with patch('translate_reports.TRANSLATOR_REGISTRY', {'qwen':translate}), \
             patch('translate_reports.cleanup_cuda'):
            out = _translate_one_translator_columns(df, 'qwen', ['terms'], 'fixture', 100, 'cpu')
        self.assertEqual(calls, [['c1','c2','r1','r2']])
        self.assertEqual(out['generation_qwen_terms'], ['en:c1','en:c2'])
        self.assertEqual(out['gt_qwen_terms'], ['en:r1','en:r2'])

    def test_pool_and_constraints(self):
        ru, text, img = u.feature_specs_from_registry(u.TRANSLATORS_DEFAULT, u.TRANSLATION_KINDS_DEFAULT)
        self.assertEqual((len(ru), len(text), len(img)), (1, 35, 5))
        candidates = u.make_candidate_configs([s.feature_col for s in ru+text+img], u.TRANSLATORS_DEFAULT,
                                               u.TRANSLATION_KINDS_DEFAULT)
        self.assertEqual(len(candidates), 35)
        self.assertTrue(all(c.text_feature.translator == c.image_feature.translator and
                            c.text_feature.kind == c.image_feature.kind for c in candidates))
        w = u.build_weight_grid(3, 0.05)
        self.assertEqual(w.shape, (190, 3))
        self.assertTrue((w[:, [0, 2]] > 0).all())
        np.testing.assert_allclose(w.sum(1), 1)
        with self.assertRaises(ValueError):
            u.build_weight_grid(3, 0.3)

    def test_signed_radcliq_is_explicit(self):
        spec = u.FeatureSpec('radcliq', 'x', 'english_text', False)
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            u._quality_feature_matrix(pd.DataFrame({'x': [-1, 0, 1]}), [spec])
        np.testing.assert_allclose(u._quality_feature_matrix(pd.DataFrame({'x': [1, 2, 4]}), [spec])[:,0], [1,.5,.25])
        spec = u.FeatureSpec('radcliq', 'x', 'english_text', False, quality_transform='negate')
        np.testing.assert_allclose(u._quality_feature_matrix(pd.DataFrame({'x': [-1, 0, 1]}), [spec])[:,0], [1,0,-1])

    def test_calibration_and_training_only_normalization(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 5, 30).astype(float)
        df = pd.DataFrame({'report_id': [f'{i:03}' for i in range(30)], 'human_score': y,
                           'ru_GREEN_Qwen': -y + rng.normal(0,.2,30),
                           'text_en_cosinesim_biovilt_qwen_terms': -y + rng.normal(0,.4,30),
                           'img_en_biovilt_qwen_terms': -y + rng.normal(0,.7,30)})
        fit_errors = []
        original = u.fit_calibration
        def spy(raw, errors):
            fit_errors.append(np.asarray(errors).copy())
            return original(raw, errors)
        with patch.object(u, 'fit_calibration', side_effect=spy):
            result = u.select_and_fit_final_config(df, 'human_score', ['qwen'], ['terms'],
                                                   k_outer=3, k_inner=3, weight_step=.25)
        self.assertEqual([len(x) for x in fit_errors], [20,20,20,30])
        for outer, errors in zip(result['nested_cv_outer_results'], fit_errors):
            test_ids = outer['test_row_ids']
            train = df[~df.report_id.isin(test_ids)]
            np.testing.assert_array_equal(errors, train.human_score)
            x = u._quality_feature_matrix(train, [u.FeatureSpec(**s) for s in outer['features']])
            np.testing.assert_allclose(outer['standardizer']['means'], x.mean(0))
            self.assertGreater(outer['calibration']['a'], 0)
            self.assertTrue(all(0 <= x <= 1 for x in outer['test_quality_scores']))
            self.assertLess(outer['kendall_tau_with_errors'], 0)
        scores = u.score_with_config(df, result)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        single = np.concatenate([u.score_with_config(df.iloc[[i]], result) for i in range(30)])
        np.testing.assert_allclose(scores, single)
        bad = df.copy(); bad.loc[0, 'ru_GREEN_Qwen'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing/non-finite'):
            u.select_and_fit_final_config(bad, 'human_score', ['qwen'], ['terms'])
        with self.assertRaisesRegex(ValueError, 'Too few'):
            u.select_and_fit_final_config(df.head(6), 'human_score', ['qwen'], ['terms'], k_outer=5, k_inner=5)

    def test_pair_pool_skips_gemma_terms(self):
        df = pd.DataFrame({'generation_translategemma_noterms': ['a'], 'gt_translategemma_noterms': ['b']})
        self.assertEqual(len(pair_columns(df, ['translategemma'], ['terms', 'noterms'])), 1)

    def test_radeval_loads_only_requested_metric(self):
        flags = []
        class FakeRadEval:
            def __init__(self, **kwargs): flags.append(kwargs)
            def __call__(self, refs, hyps):
                return {'radcliq_v1': [0.4]}
        df = pd.DataFrame({'report_id':['001'], 'c':['candidate'], 'r':['reference']})
        with patch.dict(sys.modules, {'RadEval': types.SimpleNamespace(RadEval=FakeRadEval)}), \
             patch('calculate_english_text.cleanup_cuda'):
            result = calculate_radeval(df, [('qwen','terms','c','r')], ['radcliq'])
        self.assertEqual(result.text_en_radcliq_qwen_terms.tolist(), [.4])
        self.assertFalse(flags[0]['do_radgraph'])
        self.assertFalse(flags[0]['do_ratescore'])
        self.assertTrue(flags[0]['do_radcliq'])
        self.assertTrue(flags[0]['do_per_sample'])

    def test_green_parser_refuses_incomplete(self):
        green = object.__new__(ManualGREENQwen)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            green.parse('Это хороший отчет')
        text = '[Клинически значимые ошибки]:\nКлинически значимых ошибок нет.\n' \
               '[Клинически незначимые ошибки]:\nКлинически незначимых ошибок нет.\n' \
               '[Совпадающие находки]:\n3. Три находки'
        self.assertEqual(green.parse(text).score, 1)
        with self.assertRaises(ValueError):
            green.parse(text.replace('Клинически значимых ошибок нет.', '(а) Неверная находка: 1. тест'))
        with self.assertRaisesRegex(ValueError, 'max_new_tokens'):
            require_complete_generation([3, 4, 5], 3, 2)

    def test_join_and_ids(self):
        left = pd.DataFrame({'report_id': ['001', '002'], 'a': [1, 2]})
        right = pd.DataFrame({'report_id': ['002', '001'], 'b': [4, 3]})
        self.assertEqual(p.merge_exact(left, right).b.tolist(), [3, 4])
        for bad in [pd.DataFrame({'report_id': ['001', '001']}), pd.DataFrame({'report_id': ['001']})]:
            with self.assertRaises(ValueError): p.merge_exact(left, bad)


class PipelineTests(unittest.TestCase):
    def test_environment_and_selected_jobs(self):
        plan = p.resolved_plan({'environments': {'default': 'base', 'translate.qwen': 'qwen-env'}})
        jobs = p.stages(plan)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(p.environment(plan, jobs[0]), ['conda','run','--no-capture-output','-n','qwen-env','python'])
        specs = [{'group':'ru_text','method_key':'ru_GREEN_Qwen'},
                 {'group':'english_text','method_key':'cosinesim_biovilt','translator':'qwen','kind':'terms'},
                 {'group':'image_text','method_key':'img_biovilt','translator':'qwen','kind':'terms'}]
        jobs = p.stages(plan, specs)
        self.assertEqual([j['id'] for j in jobs], ['translate.qwen.terms','green','text.biovilt.qwen.terms','image.qwen.terms'])

    def test_file_pipeline_resume_fit_and_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'image.png').write_bytes(b'fixture image bytes; neural workers are mocked')
            source = pd.DataFrame({'report_id': [f'{i:03}' for i in range(18)],
                                   'cand': ['Текст '+str(i) for i in range(18)], 'ref': ['Эталон']*18,
                                   'img': ['image.png']*18, 'errors': [i%4 for i in range(18)]})
            csv = root/'data.csv'; source.to_csv(csv,index=False)
            plan = p.resolved_plan({'input_csv': str(csv), 'image_root': str(root),
                                    'columns': {'candidate':'cand','reference':'ref','image':'img','target':'errors','row_id':'report_id'},
                                    'translators':['qwen'], 'translation_kinds':['terms'],
                                    'methods':['cosinesim_biovilt'],
                                    'environments': {'default':[sys.executable]},
                                    'runtime': {'device':'cpu'},
                                    'training': {'k_outer':3,'k_inner':3,'weight_step':.25}})
            calls = []
            def fake_worker(cmd, **kwargs):
                request = p.read_json(cmd[-1]); j = request['job']; wd = Path(request['work_dir'])
                base = p.read_csv(wd/'base.csv'); n = len(base)
                calls.append(j['id'])
                out = pd.DataFrame({'report_id': base.report_id})
                # Fixture features with no model downloads or claim of neural validation.
                ids = base.report_id.astype(int).to_numpy()
                for col in p.expected_columns(j):
                    if col in {'ru_GREEN_Qwen_analysis','ru_GREEN_Qwen_errors_json'}:
                        out[col] = 'fixture'
                    elif j['stage'] == 'translate': out[col] = 'English fixture'
                    else: out[col] = -(ids % 4) + np.sin(ids)*.1
                path, _ = p.paths(wd,j); out.to_csv(path,index=False)
                p.atomic_json(path.with_suffix('.versions.json'), {'python': 'fixture', 'packages': {}})
            wd = root/'train'
            with contextlib.redirect_stdout(io.StringIO()), patch.object(p.subprocess,'run',side_effect=fake_worker):
                p.prepare(plan,wd,True)
                features = p.run_features(plan,wd)
                self.assertEqual(len(calls),4)
                self.assertEqual(features.report_id.iloc[0],'000')
                p.run_features(plan,wd,resume=True)
                self.assertEqual(len(calls),4)
                # Fitting works entirely from cached scores, even without source images.
                (root/'image.png').unlink()
                artifact = p.fit(plan,wd,root/'metric.json')
                self.assertEqual(len(calls),4)
                (root/'image.png').write_bytes(b'fixture image bytes; neural workers are mocked')
                test_csv = root/'new.csv'; source.drop(columns='errors').head(3).to_csv(test_csv,index=False)
                out = p.infer(artifact,test_csv,root/'test',root/'scores.json')
                self.assertEqual(len(calls),8)
                self.assertEqual([r['report_id'] for r in out['entry-wise']],['000','001','002'])
                self.assertTrue(all(0 <= r['quality_score'] <= 1 for r in out['entry-wise']))
                # Altering model settings invalidates the cache before fitting.
                changed = copy.deepcopy(plan); changed['models']['green_model'] = 'another-model'
                with self.assertRaisesRegex(ValueError,'stale'):
                    p.run_features(changed,wd,cache_only=True)
                # Tampering with a completed file cannot silently enter the fit.
                target = wd/'green.csv'; target.write_text(target.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'stale'):
                    p.run_features(plan,wd,cache_only=True)
                with self.assertRaisesRegex(ValueError,'execution locations'):
                    p.infer(artifact,test_csv,root/'test2',root/'scores2.json', {'models':{}})

    def test_worker_preserves_ids_and_imports_only_selected_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = pd.DataFrame({'report_id':['001','002'], 'candidate_report':['a','b'],
                                 'reference_report':['r','s'], 'image_path':['x','y']})
            base.to_csv(root/'base.csv', index=False)
            plan = p.resolved_plan({'runtime':{'device':'cpu'}, 'environments':{'default':[sys.executable]}})
            job = {'id':'green','stage':'green','deps':[]}
            p.atomic_json(root/'job.json', {'plan':plan,'job':job,'work_dir':str(root)})
            def fake_green(**kwargs):
                df = p.read_csv(kwargs['input_csv'])
                df = df[['report_id']].assign(ru_GREEN_Qwen=[1,.5],ru_GREEN_Qwen_analysis='fixture',ru_GREEN_Qwen_errors_json='{}')
                df.to_csv(kwargs['output_csv'],index=False)
            with patch('calculate_russian_text.calculate_green_for_csv',side_effect=fake_green):
                p.worker(root/'job.json')
            self.assertEqual(p.read_csv(root/'green.csv').report_id.tolist(), ['001','002'])
            self.assertTrue((root/'green.versions.json').exists())


if __name__ == '__main__':
    unittest.main()
