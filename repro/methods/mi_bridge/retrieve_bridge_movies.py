"""Exact MovieLens genre AND retrieval, without popularity or semantic expansion."""
from pathlib import Path


def load_movies(path):
    movies = []
    for line in Path(path).read_text(encoding='ISO-8859-1').splitlines():
        movie_id, title, genres = line.split('::')
        movies.append({'id': int(movie_id), 'title': title, 'genres': genres.split('|')})
    return movies


def retrieve_bridge_movies(pair, movies, history, target):
    excluded_ids = {x['id'] for x in history} | {target['id']}
    excluded_titles = {x['title'] for x in history} | {target['title']}
    required = {pair['user_existing_interest'], pair['target_related_interest']}
    matches = sorted((m for m in movies if required.issubset(m['genres'])
                      and m['id'] not in excluded_ids and m['title'] not in excluded_titles),
                     key=lambda m: m['id'])
    return {'candidate_count': len(matches), 'saved_candidate_count': min(len(matches), 20),
            'candidates': matches[:20]}
