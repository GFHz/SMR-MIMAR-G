import ast,unittest
from pathlib import Path
from repro.methods.mimar_v01.candidate_builder import build_candidates
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence
from repro.methods.mimar_v01.feedback import ACCEPT,REJECT,apply_feedback
from repro.methods.mimar_v01.route_scoring import KAPPA
from repro.methods.mimar_v01.route_state import FeedbackMemory,ROUTE_PENALTY_DECAY,ROUTE_REJECTION_INCREMENT
from repro.methods.mimar_v02.coverage import *
from repro.methods.mimar_v02.planner import MIMARV02Planner
from repro.methods.mimar_v02.prompt import build_prompt,parse_one_title
from repro.methods.mimar_v02.route_scoring import ALPHA,BETA,DELTA_ROUTE,ETA,route_table

def m(i,t,g):return {"id":i,"title":t,"genres":g}
class V02Tests(unittest.TestCase):
 def setUp(self):
  self.movies=[m(1,"H",["Comedy"]),m(2,"Both",["Comedy","Drama"]),m(3,"Action",["Action","Drama"]),m(4,"Multi",["Drama","Sci-Fi"]),m(5,"Target",["Drama","Sci-Fi"])]
  self.co=GenreCooccurrence(self.movies);self.long={"Comedy":{"score":.5},"Action":{"score":.2}}
 def table(self,active=None,cov=None,previous=None,memory=None):return route_table(self.long,active or {"Comedy":.5},["Drama","Sci-Fi"],cov or initial_coverage(["Drama","Sci-Fi"]),self.co,memory or FeedbackMemory(),previous)
 def test_01_c_correct(self):self.assertAlmostEqual(self.co.score("Comedy","Drama"),1/(2*4)**.5)
 def test_02_c_not_additive(self):
  row=self.table()[0];self.assertAlmostEqual(row["route_score"],ALPHA*row["long_term"]+BETA*row["active"]+ETA*row["need"]-DELTA_ROUTE*row["penalty"]+row["continuity_bonus"])
 def test_03_nonzero_feasible(self):self.assertTrue(next(x for x in self.table() if x["target_genre"]=="Drama")["feasible"])
 def test_04_zero_infeasible(self):self.assertFalse(next(x for x in self.table() if x["target_genre"]=="Sci-Fi")["feasible"])
 def test_05_initial_coverage(self):self.assertEqual(initial_coverage(["Drama","Sci-Fi"]),{"Drama":0.0,"Sci-Fi":0.0})
 def test_06_initial_need(self):self.assertEqual(target_need(initial_coverage(["Drama"])),{"Drama":1.0})
 def test_07_single_update(self):self.assertEqual(update_coverage(initial_coverage(["Drama","Sci-Fi"]),["Drama"])[0],{"Drama":.5,"Sci-Fi":0})
 def test_08_multi_update(self):self.assertEqual(update_coverage(initial_coverage(["Drama","Sci-Fi"]),["Drama","Sci-Fi"])[0],{"Drama":.5,"Sci-Fi":.5})
 def test_09_clipped(self):
  x={"Drama":.75};self.assertEqual(update_coverage(x,["Drama"])[0]["Drama"],1)
 def test_10_need_identity(self):
  c={"Drama":.5};self.assertEqual(target_need(c)["Drama"],1-c["Drama"])
 def test_11_need_changes_target_winner(self):
  co=GenreCooccurrence(self.movies+[m(6,"Comedy Sci-Fi",["Comedy","Sci-Fi"])])
  before=route_table(self.long,{"Comedy":.5},["Drama","Sci-Fi"],{"Drama":0,"Sci-Fi":0},co,FeedbackMemory())
  after=route_table(self.long,{"Comedy":.5},["Drama","Sci-Fi"],{"Drama":1,"Sci-Fi":0},co,FeedbackMemory())
  self.assertEqual(before[0]["target_genre"],"Drama");self.assertEqual(after[0]["target_genre"],"Sci-Fi")
 def test_12_new_active_creates_routes(self):
  rows=self.table(active={"Comedy":.5,"Action":.1});self.assertTrue(any(x["interest"]=="Action" for x in rows))
 def test_13_new_active_can_win(self):
  rows=self.table(active={"Comedy":.1,"Action":1});self.assertEqual(rows[0]["interest"],"Action")
 def test_14_continuity_unchanged(self):self.assertEqual(KAPPA,.02)
 def test_15_feedback_penalty_unchanged(self):self.assertEqual((ROUTE_PENALTY_DECAY,ROUTE_REJECTION_INCREMENT),(.8,.5))
 def test_16_candidate_exclusions(self):
  rows=build_candidates(self.movies,("Comedy","Drama"),{"Comedy":.5},self.co,{1},{2},5,FeedbackMemory());self.assertFalse({1,2,5}&{x["id"] for x in rows})
 def test_17_prompt_route_labels(self):
  row=self.table()[0];p=build_prompt({},self.long,{"Comedy":.5},self.movies[-1],initial_coverage(["Drama","Sci-Fi"]),{"Drama":1,"Sci-Fi":1},row,[self.movies[1]])
  self.assertIn("ABSTRACT GENRE CONCEPTS",p["system_prompt"]);self.assertTrue(p["user_prompt"].endswith("list above."))
 def test_18_exact_parser(self):
  self.assertTrue(parse_one_title("Both",["Both"])["success"]);self.assertFalse(parse_one_title("Comedy",["Both"])["success"])
 def test_19_no_evaluator_import(self):
  root=Path(__file__).resolve().parents[1]
  for p in root.glob("*.py"):
   imports=" ".join(ast.unparse(n).lower() for n in ast.walk(ast.parse(p.read_text(encoding="utf8"))) if isinstance(n,(ast.Import,ast.ImportFrom)))
   self.assertNotIn("sasrec",imports);self.assertNotIn("evaluators.formal",imports)
 def test_20_planner_coverage_stop(self):
  history=[m(i,f"H{i}",["Comedy"]) for i in range(10,30)];planner=MIMARV02Planner(history,m(5,"Target",["Drama"]),history+self.movies)
  planner.coverage={"Drama":1};self.assertEqual(planner.step(lambda p:"Both",lambda i,r:ACCEPT)["status"],"target_coverage_complete")
 def test_21_step_logging_identity_and_state(self):
  history=[m(i,f"H{i}",["Comedy"]) for i in range(10,30)]
  planner=MIMARV02Planner(history,m(5,"Target",["Drama"]),history+self.movies,user_id=7,path_id=2)
  row=planner.step(lambda p:"Both",lambda i,r:ACCEPT)
  self.assertEqual((row["user_id"],row["path_id"],row["step"]),(7,2,1))
  self.assertIn("long_term_interests",row);self.assertEqual(row["target_genres"],["Drama"])
if __name__=="__main__":unittest.main()
