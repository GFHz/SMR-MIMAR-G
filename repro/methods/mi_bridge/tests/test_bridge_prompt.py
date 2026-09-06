import json
import unittest
from unittest.mock import patch
from repro.experiments.case_study.run_mi_bridge_v1_case import load_inputs, BASE
from repro.experiments.case_study.compare_mi_bridge_v1_case import analyze_path
from repro.methods.mi_bridge.bridge_prompt import build_bridge_prompt
from repro.utils import ROOT,read_json


class BridgePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('repro.methods.mi_bridge.find_fallback_bridge.find_fallback_bridge',side_effect=AssertionError('Fallback forbidden')):
            cls.sample,cls.baseline,cls.cfg,cls.selected,cls.saved,cls.ours=load_inputs()
    def test_baseline_original_unchanged(self):
        self.assertEqual(self.ours['system_prompt'],self.baseline['system_prompt'])
        self.assertEqual(self.ours['user_prompt'],self.baseline['user_prompt']+self.ours['bridge_context'])
    def test_exactly_one_block(self):
        self.assertEqual(self.ours['user_prompt'].count('[BRIDGE CONTEXT]'),1)
        with self.assertRaises(ValueError):build_bridge_prompt('',self.ours['user_prompt'],self.saved['selected_bridge_pair'],'direct',[])
    def test_selected_from_saved_result(self):
        source=read_json(ROOT/'repro/results/case_study/user_1_bridge_selection.json')
        self.assertEqual(self.selected['selected_bridge'],source['selected_bridge_pair'])
        self.assertEqual(self.selected['direct_candidate_count'],6)
    def test_candidates_from_saved_result(self):
        line=next(x for x in self.ours['bridge_context'].splitlines() if x.startswith('["'))
        self.assertEqual(json.loads(line),[x['title'] for x in self.saved['bridge_movie_candidates']])
    def test_no_target_last_constraint(self):
        self.assertNotIn('last',self.ours['bridge_context'].lower())
        self.assertNotIn('must',self.ours['bridge_context'].lower())
    def test_no_history_exclusion_constraint(self):
        text=self.ours['bridge_context'].lower()
        self.assertNotIn('not use',text);self.assertNotIn('never',text);self.assertNotIn('exclude',text)
    def test_no_fixed_position(self):
        text=self.ours['bridge_context'].lower()
        self.assertNotIn('step',text);self.assertNotIn('position',text);self.assertNotIn('exactly',text)
    def test_no_full_pair_every_step(self):
        self.assertNotIn('every',self.ours['bridge_context'].lower())
        self.assertIn('you may use',self.ours['bridge_context'])
    def test_sample_unchanged(self):
        self.assertEqual(self.sample,read_json(BASE/'sample.json'))
    def test_config_unchanged(self):
        self.assertEqual(self.cfg,read_json(BASE/'run_metadata.json')['configuration'])
    def test_direct_only(self):
        with self.assertRaises(ValueError):build_bridge_prompt('','',self.saved['selected_bridge_pair'],'multi_hop',[])
    def test_metrics_exclude_target_and_preserve_invalid(self):
        sample={'target_title':'T','history':[{'title':'H'}]}
        catalog={'T':[{'genres':['I','G']}],'H':[{'genres':['I']}],'B':[{'genres':['I','G']}]}
        r=analyze_path({'path':['T','H','B','B','Unknown'],'parse_success':True},sample,catalog,[{'title':'B'}],'I','G')
        m=r['metrics'];p=r['positions']
        self.assertEqual(m['history_reuse_rate'],1/4)
        self.assertEqual(m['new_intermediate_item_count'],3)
        self.assertEqual(m['full_bridge_hit_rate'],2/4)
        self.assertEqual(m['bridge_candidate_usage_count'],2)
        self.assertEqual((m['invalid_item_count'],m['duplicate_item_count']),(1,1))
        self.assertEqual((p['full_bridge_position'],p['existing_interest_last_position']),(3,4))
    def test_empty_rates(self):
        r=analyze_path({'path':[],'parse_success':False},{'target_title':'T','history':[]},{},[],'I','G')
        self.assertIsNone(r['metrics']['full_bridge_hit_rate'])
        self.assertIsNone(r['positions']['full_bridge_position'])


if __name__=='__main__':unittest.main()
