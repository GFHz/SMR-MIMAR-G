"""Exact catalog lookup plus article-only normalization; no fuzzy or LLM repairs."""
import re
from collections import defaultdict
from pathlib import Path


def article_key(title):
    """Keep every non-article word, case, punctuation and four-digit year intact.

    Accept one leading/trailing The/A/An, or the same article redundantly at both
    ends. Conflicting articles, missing year or repeated extra articles do not get
    repaired. Retaining the article in the key prevents matching article-less titles.
    """
    match = re.fullmatch(r'(.+) \((\d{4})\)', title)
    if not match:
        return None
    body, year = match.groups()
    leading = re.match(r'^(The|A|An) (.+)$', body)
    trailing = re.fullmatch(r'(.+), (The|A|An)', body)
    if not leading and not trailing:
        return None
    if trailing:
        stem, article = trailing.groups()
        if leading:
            if leading.group(1) != article:
                return None
            stem = stem[len(article) + 1:]
    else:
        article, stem = leading.groups()
    return (stem, article, year) if stem else None


class TitleResolver:
    def __init__(self, movies):
        self.movies = {int(movie['id']): dict(movie) for movie in movies}
        self.exact = defaultdict(list)
        self.articles = defaultdict(list)
        for movie in self.movies.values():
            self.exact[movie['title']].append(movie)
            key = article_key(movie['title'])
            if key is not None:
                self.articles[key].append(movie)

    @classmethod
    def from_movies_dat(cls, path):
        movies = []
        with Path(path).open(encoding='latin-1') as stream:
            for line in stream:
                item_id, title, genres = line.rstrip('\r\n').split('::')
                movies.append({'id': int(item_id), 'title': title, 'genres': genres.split('|')})
        return cls(movies)

    def resolve(self, raw_title):
        if not isinstance(raw_title, str):
            raise TypeError('Resolver only accepts strings from parsed paths')
        candidates = self.exact.get(raw_title, [])
        kind = 'exact'
        if not candidates:
            kind = 'article_normalization'
            key = article_key(raw_title)
            candidates = self.articles.get(key, []) if key is not None else []
        candidates = sorted(candidates, key=lambda x: x['id'])
        resolved = candidates[0] if len(candidates) == 1 else None
        return {'raw_title': raw_title, 'resolved_title': resolved['title'] if resolved else None,
                'raw_item_id': resolved['id'] if resolved else None,
                'genres': list(resolved['genres']) if resolved else None,
                'resolution_status': 'resolved' if resolved else ('ambiguous' if candidates else 'unresolved'),
                'resolution_type': kind if candidates else 'none',
                'candidate_count': len(candidates),
                'candidate_titles': [x['title'] for x in candidates]}
