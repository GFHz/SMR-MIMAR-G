import ast,unittest
from pathlib import Path
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence
from repro.methods.smr_mimar_v01.routes import *
from repro.methods.smr_mimar_v01.candidates import *
from repro.methods.smr_mimar_v01.prompt import build_prompt,parse_one_title,MAX_LLM_RETRIES
from repro.methods.smr_mimar_v01.planner import SMRMIMARPlanner

def m(i,t,g):return {"id":i,"title":t,"genres":g}
class SMRTests(unittest.TestCase):
 def setUp(self):
  self.history=[m(i,f"H{i}",["Comedy"] if i<15 else ["Drama"]) for i in range(1,21)]
  self.catalog=self.history+[m(30,"CD",["Comedy","Drama"]),m(31,"CS",["Comedy","Sci-Fi"]),m(32,"DS",["Drama","Sci-Fi"]),m(33,"CDS",["Comedy","Drama","Sci-Fi"]),m(40,"Target",["Drama","Sci-Fi"])]
  self.co=GenreCooccurrence(self.catalog);self.top,self.profile=ranked_long_term_interests(self.history)
  self.routes=feasible_routes(self.top,["Drama","Sci-Fi"],self.co);self.selected=select_diverse_routes(self.routes)
 def test_01_full_history_profile(self):self.assertEqual(self.profile["Comedy"]["count"],14)
 def test_02_no_short_term_state(self):self.assertFalse(hasattr(SMRMIMARPlanner(self.history,self.catalog[-1],self.catalog),"active"))
 def test_03_all_target_genres(self):self.assertEqual({x["target_genre"] for x in self.routes},{"Drama","Sci-Fi"})
 def test_04_cooccurrence(self):self.assertGreater(self.co.score("Comedy","Sci-Fi"),0)
 def test_05_zero_support_infeasible(self):self.assertFalse(any(x["interest"]=="Drama" and x["target_genre"]=="Comedy" for x in feasible_routes(self.top,["Comedy"],GenreCooccurrence(self.history))))
 def test_06_identity_feasible(self):self.assertTrue(any(x["interest"]=="Drama" and x["target_genre"]=="Drama" for x in self.routes))
 def test_07_route_prior_formula(self):
  x=self.routes[0];self.assertAlmostEqual(x["route_prior"],ALPHA*x["L"]+GAMMA*x["C"])
 def test_08_route_set_static(self):
  p=SMRMIMARPlanner(self.history,self.catalog[-1],self.catalog);before=p.route_set;p.step(lambda q:q["allowed_titles"][0]);self.assertEqual(before,p.route_set)
 def test_09_diversity_deterministic(self):self.assertEqual(self.selected,select_diverse_routes(self.routes))
 def test_10_multiple_starts(self):self.assertGreaterEqual(len({x["interest"] for x in self.selected}),2)
 def test_11_multiple_targets(self):self.assertEqual({x["target_genre"] for x in self.selected},{"Drama","Sci-Fi"})
 def test_12_multitag_multi_support(self):self.assertGreaterEqual(len(supported_routes(self.catalog[-2],self.selected)),2)
 def test_13_history_excluded(self):self.assertFalse({x["id"] for x in self.history}&{x["id"] for x in legal_candidates(self.catalog,self.selected,{x["id"] for x in self.history},set(),40)})
 def test_14_path_duplicate_excluded(self):self.assertNotIn(30,{x["id"] for x in legal_candidates(self.catalog,self.selected,set(),{30},40)})
 def test_15_target_excluded(self):self.assertNotIn(40,{x["id"] for x in legal_candidates(self.catalog,self.selected,set(),set(),40)})
 def test_16_prompt_full_routes(self):self.assertIn("static_selected_route_set",build_prompt({},self.top,self.catalog[-1],self.selected,[],legal_candidates(self.catalog,self.selected,{x["id"] for x in self.history},set(),40))["user_prompt"])
 def test_17_prompt_labels_not_titles(self):self.assertIn("ROUTE LABELS ARE GENRE CONCEPTS, NOT MOVIE TITLES",build_prompt({},self.top,self.catalog[-1],self.selected,[],legal_candidates(self.catalog,self.selected,set(),set(),40))["system_prompt"])
 def test_18_exact_parser(self):self.assertTrue(parse_one_title("CD",["CD"])["success"]);self.assertFalse(parse_one_title("Comedy",["CD"])["success"])
 def test_19_retry_limit(self):self.assertEqual(MAX_LLM_RETRIES,2)
 def test_20_no_dynamic_update(self):
  p=SMRMIMARPlanner(self.history,self.catalog[-1],self.catalog);before=p.long_term_profile;p.step(lambda q:q["allowed_titles"][0]);self.assertEqual(before,p.long_term_profile)
 def test_21_no_coverage_need(self):
  p=SMRMIMARPlanner(self.history,self.catalog[-1],self.catalog);self.assertFalse(hasattr(p,"coverage"));self.assertFalse(hasattr(p,"need"))
 def test_22_no_switch_commitment_penalty(self):
  p=SMRMIMARPlanner(self.history,self.catalog[-1],self.catalog)
  for name in ("previous_route","commitments","memory"):self.assertFalse(hasattr(p,name))
 def test_23_no_sasrec_import(self):
  root=Path(__file__).resolve().parents[1]
  for p in root.glob("*.py"):
   imports=" ".join(ast.unparse(n).lower() for n in ast.walk(ast.parse(p.read_text(encoding="utf8"))) if isinstance(n,(ast.Import,ast.ImportFrom)));self.assertNotIn("sasrec",imports)
 def test_24_six_step_budget(self):
  from repro.methods.smr_mimar_v01.planner import MAX_INTERMEDIATE_STEPS
  self.assertEqual(MAX_INTERMEDIATE_STEPS,6)
 def test_25_prompt_simultaneous_route_contract(self):
  from repro.methods.smr_mimar_v01.prompt import SYSTEM_PROMPT
  for text in ("The routes are simultaneous migration directions, not a single route to follow.","A candidate may support multiple routes at the same time.","You do not need to select one route before selecting the movie."):
   self.assertIn(text,SYSTEM_PROMPT)
if __name__=="__main__":unittest.main()
