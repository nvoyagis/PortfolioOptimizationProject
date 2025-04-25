import unittest
from graph import *

# NOTE conn_components() doesn't work as intended
class TestList(unittest.TestCase):

    def test_01(self) -> None:
        g = Graph('test1.txt')
        self.assertEqual([['v1', 'v2', 'v3', 'v4', 'v5'], ['v6', 'v7', 'v8', 'v9']], g.conn_components())
        self.assertTrue(g.is_bipartite())
    
    def test_02(self) -> None:
        g = Graph('test2.txt')
        self.assertEqual([['v1', 'v2', 'v3'], ['v4', 'v6', 'v7', 'v8']], g.conn_components())
        self.assertFalse(g.is_bipartite())

    def test_03(self) -> None:
        g = Graph('test3.txt')
        self.assertEqual([['v1', 'v2']], g.conn_components())
        self.assertTrue(g.is_bipartite())

    def test_04(self) -> None:
        g = Graph('test4.txt')
        self.assertEqual([['v1', 'v2', 'v3', 'v4', 'v5']], g.conn_components())
        self.assertFalse(g.is_bipartite())

    def test_05(self) -> None:
        g = Graph('test5.txt')
        self.assertEqual([['1', '2', '3', '4', '5'], ['6', '7', '8', '9']], g.conn_components())
        self.assertTrue(g.is_bipartite())

    def test_06(self) -> None:
        g = Graph('test6.txt')
        self.assertFalse(g.is_bipartite())

    def test_07(self) -> None:
        g = Graph('test7.txt')
        self.assertEqual([['v1', 'v2', 'v3', 'v4', 'v5', 'v6'], ['v10', 'v11', 'v12', 'v13', 'v14', 'v15']], g.conn_components())
        self.assertTrue(g.is_bipartite())

if __name__ == '__main__':
   unittest.main()