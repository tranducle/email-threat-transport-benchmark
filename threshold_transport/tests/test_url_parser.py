"""Regression tests: the SpaPhish URL parser must return the URLs of the bracketed format."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "external_corpora"))
from url_parser import parse_urls, url_features  # noqa: E402


class SpaPhishFormat(unittest.TestCase):
    def test_bracketed_list(self):
        raw = "[http://example.com/aB1c.asp?dX5],[https://img.example.org/Vb0e.png],[https://img.example.org/X3dz.png]"
        self.assertEqual(parse_urls(raw), ["http://example.com/aB1c.asp?dX5",
                                           "https://img.example.org/Vb0e.png",
                                           "https://img.example.org/X3dz.png"])

    def test_placeholder_items_dropped(self):
        raw = "[https://shop.example.net/soporte/corte/],[#],[https://files.example.pl/a.png?ts=1]"
        self.assertEqual(len(parse_urls(raw)), 2)

    def test_mixed_case_scheme(self):
        raw = "[hTtps://storage.example.com/v5b4/page99.html#LIP],[https://fonts.example.com/css?family=Oswald]"
        self.assertEqual(len(parse_urls(raw)), 2)

    def test_commas_inside_url_kept(self):
        raw = "[https://fonts.example.com/css?family=Open+Sans:400,400i,700,700i],[http://a.example.org/x]"
        self.assertEqual(parse_urls(raw)[0], "https://fonts.example.com/css?family=Open+Sans:400,400i,700,700i")

    def test_development_quoted_list_unchanged(self):
        self.assertEqual(parse_urls("['http://www.example.ie/mailman/listinfo/list']"),
                         ["http://www.example.ie/mailman/listinfo/list"])

    def test_empty(self):
        for v in (None, "", "[]", "nan"):
            self.assertEqual(parse_urls(v), [])

    def test_features_nonzero(self):
        f = url_features("[https://a.b.example.com/path],[http://x.org]")
        self.assertEqual(f["url_indicator"], 1)
        self.assertEqual(f["url_count_parsed"], 2)
        self.assertEqual(f["url_subdom_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
