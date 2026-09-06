import unittest
from fractions import Fraction
from repro.methods.mi_bridge.select_bridge import bridge_score, target_need, generate_bridge_pairs, rank_bridge_pairs, select_bridge


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [{'user_existing_interest': 'i'+str(i), 'target_related_interest':'g', 'candidate_count':1} for i in range(1,4)]
    def test_score(self):
        self.assertEqual(bridge_score('Children\'s','Romance',{"Children's":13,'Romance':1},20),Fraction(8,5))
    def test_target_need(self):
        self.assertEqual(target_need('Drama',{'Drama':5},20),Fraction(3,4))
        self.assertEqual(target_need('Unknown',{},20),1)
    def test_same_genre_excluded(self):
        a={'history':[{}]*20,'genre_statistics':[{'genre':'Drama','count':5}], 'top_interests':[{'genre':'Drama'}], 'target':{'genres':['Drama','Romance']}}
        p=generate_bridge_pairs(a)
        self.assertEqual(len(p),1)
        self.assertEqual(p[0]['target_related_interest'],'Romance')
    def test_deterministic_all_tiebreakers(self):
        pairs=[{'bridge_score':s,'user_frequency':f,'user_existing_interest':i,'target_related_interest':g} for s,f,i,g in [(1,.3,'B','Z'),(1,.3,'A','Z'),(1,.3,'A','Y'),(1,.4,'C','Z'),(2,.1,'D','Z')]]
        ranked=rank_bridge_pairs(pairs)
        self.assertEqual(ranked,rank_bridge_pairs(list(reversed(pairs))))
        self.assertEqual([(p['user_existing_interest'],p['target_related_interest']) for p in ranked],[('D','Z'),('C','Z'),('A','Y'),('A','Z'),('B','Z')])
    def test_default_first(self):
        self.assertEqual(select_bridge(self.pairs),self.pairs[0])
    def test_zero_candidate_fallback(self):
        self.pairs[0]['candidate_count']=0
        self.assertEqual(select_bridge(self.pairs),self.pairs[1])
    def test_reject_b1_returns_b2(self):
        self.assertEqual(select_bridge(self.pairs,[self.pairs[0]]),self.pairs[1])
    def test_reject_b1_b2_returns_b3(self):
        self.assertEqual(select_bridge(self.pairs,self.pairs[:2]),self.pairs[2])
    def test_tuple_set_rejections(self):
        self.assertEqual(select_bridge(self.pairs,{('i1','g')}),self.pairs[1])
    def test_exhaustion(self):
        self.assertIsNone(select_bridge(self.pairs,self.pairs))
        self.assertIsNone(select_bridge([]))
        for p in self.pairs:p['candidate_count']=0
        self.assertIsNone(select_bridge(self.pairs))


if __name__=='__main__': unittest.main()
