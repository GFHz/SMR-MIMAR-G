import ast,unittest
from pathlib import Path
from repro.methods.smr_mimar_g.guard import *
from repro.methods.smr_mimar_g.planner import SMRMIMARGPlanner
from repro.methods.smr_mimar_v01.planner import SMRMIMARPlanner

def m(i,t,g):return {"id":i,"title":t,"genres":g}
class GuardTests(unittest.TestCase):
 def setUp(self):
  self.history=[m(i,f"H{i}",["Comedy"] if i<15 else ["Drama"]) for i in range(1,21)]
  self.target=m(40,"Target",["Action","Sci-Fi"]);self.catalog=self.history+[m(30,"Full",["Comedy","Action","Sci-Fi"]),m(31,"Half",["Comedy","Action"]),m(32,"Full2",["Drama","Action","Sci-Fi"]),self.target]
 def planner(self):return SMRMIMARGPlanner(self.history,self.target,self.catalog)
 def test_01_first_bypass(self):self.assertTrue(overlap_guard(None,.5)["first_item_bypass"])
 def test_02_equal_allowed(self):self.assertFalse(overlap_guard(.5,.5)["guard_triggered"])
 def test_03_increase_allowed(self):self.assertFalse(overlap_guard(.5,1)["guard_triggered"])
 def test_04_decrease_stops(self):self.assertTrue(overlap_guard(1,.5)["guard_triggered"])
 def test_05_decreased_item_not_added(self):
  p=self.planner();p.step(lambda q:"Full");p.step(lambda q:"Half");self.assertEqual([x["title"] for x in p.accepted],["Full"])
 def test_06_target_appended_after_stop(self):
  p=self.planner();p.step(lambda q:"Full");p.step(lambda q:"Half");self.assertEqual([x["title"] for x in p.snapshot()["final_path"]],["Full","Target"])
 def test_07_no_retry_after_guard(self):
  p=self.planner();calls=[0]
  def choose(q):calls[0]+=1;return "Full" if calls[0]==1 else "Half"
  p.step(choose);r=p.step(choose);self.assertEqual(calls[0],2);self.assertEqual(r["llm_retry_count"],0)
 def test_08_no_feedback_penalty(self):self.assertFalse(hasattr(self.planner(),"memory"))
 def test_09_no_dynamic_interest(self):self.assertFalse(hasattr(self.planner(),"active"))
 def test_10_static_route_prior_unchanged(self):
  a=SMRMIMARPlanner(self.history,self.target,self.catalog);b=self.planner();self.assertEqual((a.all_feasible_routes,a.route_set),(b.all_feasible_routes,b.route_set))
 def test_11_no_sasrec_import(self):
  root=Path(__file__).resolve().parents[1]
  for p in root.glob("*.py"):
   imports=" ".join(ast.unparse(n).lower() for n in ast.walk(ast.parse(p.read_text(encoding="utf8"))) if isinstance(n,(ast.Import,ast.ImportFrom)));self.assertNotIn("sasrec",imports)
 def test_12_overlap_formula(self):self.assertEqual(target_overlap(["Action","Comedy"],["Action","Sci-Fi"]),.5)
if __name__=="__main__":unittest.main()
