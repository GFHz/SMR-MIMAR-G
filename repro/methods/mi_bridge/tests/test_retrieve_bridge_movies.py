import unittest
from repro.methods.mi_bridge.retrieve_bridge_movies import retrieve_bridge_movies


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.pair={'user_existing_interest':'Animation','target_related_interest':'Romance'}
        self.movies=[{'id':i,'title':str(i),'genres':['Animation','Romance']} for i in range(30,0,-1)]
    def test_history_and_target_excluded(self):
        r=retrieve_bridge_movies(self.pair,self.movies,[self.movies[-1]],self.movies[-2])
        ids=[m['id'] for m in r['candidates']]
        self.assertNotIn(1,ids)
        self.assertNotIn(2,ids)
        self.assertEqual(r['candidate_count'],28)
    def test_count_before_cap_and_sorted(self):
        r=retrieve_bridge_movies(self.pair,self.movies,[],{'id':999,'title':'target'})
        self.assertEqual(r['candidate_count'],30)
        self.assertEqual(r['saved_candidate_count'],20)
        self.assertEqual([m['id'] for m in r['candidates']],list(range(1,21)))
    def test_and_not_or(self):
        m=[{'id':1,'title':'one','genres':['Animation']},{'id':2,'title':'two','genres':['Romance']}]
        self.assertEqual(retrieve_bridge_movies(self.pair,m,[],{'id':999,'title':'target'})['candidate_count'],0)
    def test_input_not_mutated(self):
        import copy
        before=copy.deepcopy(self.movies)
        retrieve_bridge_movies(self.pair,self.movies,[],{'id':999,'title':'target'})
        self.assertEqual(self.movies,before)


if __name__=='__main__': unittest.main()
