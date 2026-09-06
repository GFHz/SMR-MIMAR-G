import unittest
from unittest.mock import patch
from repro.methods.mi_bridge.select_bridge import BridgeRanking, select_bridge
from repro.methods.mi_bridge.find_fallback_bridge import find_fallback_bridge, rank_fallbacks, find_real_example
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.utils import ROOT, read_json


def movie(mid, genres):
    return {'id': mid, 'title': str(mid), 'genres': genres}


def pair(i='Children\'s', g='Romance'):
    return {'user_existing_interest': i, 'target_related_interest': g}


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.movies = [movie(1,["Children's",'Drama']), movie(2,['Drama','Romance'])]
        self.target = movie(99,['Romance'])
    def select(self, movies=None, pairs=None, rejected=None, history=None):
        return select_bridge(BridgeRanking([pair()] if pairs is None else pairs,
                             self.movies if movies is None else movies, history or [], self.target), rejected)
    def test_direct_never_calls_fallback(self):
        with patch('repro.methods.mi_bridge.find_fallback_bridge.find_fallback_bridge', side_effect=AssertionError('fallback called')):
            r=self.select(self.movies+[movie(3,["Children's",'Romance'])])
        self.assertEqual(r['bridge_type'],'direct')
    def test_direct_gate_on_fallback_function(self):
        r=find_fallback_bridge("Children's",'Romance',self.movies+[movie(3,["Children's",'Romance'])],[],self.target)
        self.assertEqual(r['reason'],'direct_available')
    def test_one_intermediate_and_endpoints_excluded(self):
        r=self.select()
        self.assertEqual(r['bridge_type'],'multi_hop')
        self.assertEqual(r['selected_fallback']['intermediate_interest'],'Drama')
        for x in r['ranked_fallbacks']:
            self.assertNotIn(x['intermediate_interest'],[x['existing_interest'],x['target_interest']])
    def test_both_sides_required(self):
        self.assertEqual(self.select(self.movies[:1])['bridge_type'],'unavailable')
        self.assertEqual(self.select(self.movies[1:])['bridge_type'],'unavailable')
    def test_history_excluded(self):
        self.assertEqual(self.select(history=[self.movies[0]])['bridge_type'],'unavailable')
    def test_target_excluded(self):
        self.target=self.movies[1]
        self.assertEqual(self.select()['bridge_type'],'unavailable')
    def test_no_invented_genres(self):
        r=find_fallback_bridge('invented','Romance',self.movies,[],self.target)
        self.assertIsNone(r['selected_fallback'])
    def test_sort_all_tiebreakers(self):
        entries=[{'intermediate_interest':m,'left_candidate_count':l,'right_candidate_count':r,'fallback_score':min(l,r)} for m,l,r in [('Z',2,2),('B',2,4),('A',2,4),('C',3,3)]]
        self.assertEqual([x['intermediate_interest'] for x in rank_fallbacks(entries)],['C','A','B','Z'])
        self.assertEqual(rank_fallbacks(entries),rank_fallbacks(list(reversed(entries))))
    def test_candidate_order_cap_total(self):
        movies=[movie(i,["Children's",'Drama']) for i in range(30,0,-1)]+[movie(i,['Drama','Romance']) for i in range(60,30,-1)]
        r=self.select(movies)['selected_fallback']
        self.assertEqual((r['left_candidate_count'],r['right_candidate_count']),(30,30))
        self.assertEqual([x['id'] for x in r['left_candidates']],list(range(1,21)))
        self.assertEqual([x['id'] for x in r['right_candidates']],list(range(31,51)))
    def test_skip_unavailable_pair_then_direct(self):
        r=self.select(pairs=[pair('War','Romance'),pair('Drama','Romance')])
        self.assertEqual((r['bridge_type'],r['existing_interest']),('direct','Drama'))
    def test_rejected_then_direct(self):
        pairs=[pair(),pair('Drama','Romance')]
        r=self.select(pairs=pairs,rejected=[pairs[0]])
        self.assertEqual(r['existing_interest'],'Drama')
        self.assertEqual(r['bridge_type'],'direct')
    def test_rejected_then_multihop(self):
        pairs=[pair('Drama','Romance'),pair()]
        r=self.select(pairs=pairs,rejected=[pairs[0]])
        self.assertEqual(r['bridge_type'],'multi_hop')
        self.assertEqual(r['existing_interest'],"Children's")
    def test_unavailable_empty_all_rejected(self):
        self.assertEqual(self.select(pairs=[])['bridge_type'],'unavailable')
        r=self.select(rejected=[pair()])
        self.assertEqual(r['bridge_type'],'unavailable')
        self.assertIsNone(r['selected_bridge'])
    def test_high_rank_fallback_before_lower_direct(self):
        self.assertEqual(self.select(pairs=[pair(),pair('Drama','Romance')])['bridge_type'],'multi_hop')
    def test_real_user1_and_real_fallback_case(self):
        a=read_json(ROOT/'repro/results/case_study/user_1_interest_analysis.json')
        old=read_json(ROOT/'repro/results/case_study/user_1_bridge_selection.json')
        movies=load_movies(ROOT/'dataset/ml-1m/movies.dat')
        r=select_bridge(BridgeRanking(old['ranked_bridge_pairs'],movies,a['history'],a['target']))
        self.assertEqual(r['bridge_type'],'direct')
        self.assertEqual((r['existing_interest'],r['target_interest']),("Children's",'Romance'))
        self.assertEqual(r['direct_candidate_count'],6)
        self.assertEqual(r['direct_candidates'],old['bridge_movie_candidates'])
        example=find_real_example(movies,a['history'],a['target'],[x['genre'] for x in a['top_interests']])
        self.assertIsNotNone(example)
        self.assertEqual(example['direct_candidate_count'],0)
        f=example['selected_fallback']
        excluded={x['id'] for x in a['history']}|{a['target']['id']}
        for side,endpoint in [('left',f['existing_interest']),('right',f['target_interest'])]:
            self.assertGreater(f[side+'_candidate_count'],0)
            self.assertTrue(all(x['id'] not in excluded and {endpoint,f['intermediate_interest']}<=set(x['genres']) for x in f[side+'_candidates']))


if __name__=='__main__': unittest.main()
