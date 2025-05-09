""" test_assembler_matrix.py
"""
import unittest

from draftsman.blueprintable import get_blueprintable_from_string

from src.representation.factory import Factory
from src.representation.matrix import (make_pole_matrix,
                                       make_assembler_matrix,
                                       make_belt_matrix,
                                       make_inserter_matrix,
                                       center_in_N,
                                       bound_bp_by_json)

# One assembler, one splitter, one inserter, 1 belt, and 2 ug belts.
threeassemblers = "0eJyFkdFOwzAMRf/Fzyla06HRPPIbCKG0M8VS41aJi6iq/DveOqEOxPaU2LLv8bUXaPoJx0gs4BagduAE7mWBRB37/pSTeURwQIIBDLAPp8inhKHpibsi+PaDGIsSsgHiI36BK/OrAWQhIVz1/u+zqnqkiK3QwOB2BsYh0RosoGLF7uHRwLx+8kV3fuMpNBiVZUCbaR2SE0bRbDa3kCX8hVTXkOuJfiHtBjmp49jFQd+iwV7uoKt7bu1Ns9WG/O6TFBI9p3GIcoHr3s+XcpvDGvjEmM4I+1TuD7U97G1dV7Wurvfap9XPP9U5fwMq6a/3"
simple_bp_str = "0eJyVktFugzAMRf/Fz0kFFNrB435jmqZArS5ScKIkTEOIf58pHUKj3danKI59zpXiAWrTofOaIlQD6MZSgOplgKDPpMxUi71DqEBHbEEAqXa6qRCwrY2ms2xV864JZQqjAE0n/IQqHV8FIEUdNc683+YEOBu41dLk43GZ7goBPVTJrmDoSXts5ufkiu3fqGtr9KwSwK96zkgBfeTqKBbjUttYknuSw0aSrYDRKwrO+ihrNHGLfSD7foUNzuh4M2d2waV/4/IVruOP8Gdv+byTU2bXoHKDZuP3r5PrptkfpuIh039Etos3TYdpky67V61WVcAH+jAjntL8WGbHPCvLfcnbYBSn4O7npXscvwCGe/P8"
fast_bp_str = "0eJyVksFugzAMQP/F56QqFNqRY39jmqZAvc4SGJSYaQjx7wulRR3r2u0UJbHfc2L3kJctNo5YwPRARc0ezHMPno5sy/FMugbBAAlWoIBtNe6s91jlJfFRV7Z4J0YdwaCA+ICfYKLhRQGykBBOvN/z4kA9kMNCqGYwawVN7Wna9BBgOlqlCrpwtUqHM7Z75bbK0QWVgpBLU43s0Uk4HdRsfLNetDjLvqmd6BxLeSS864uXbN+UJKP0ATU+MaNbzM2S2YZvdEdXh/VS8fJT4nOReiJeqYPo0jNu2jF3oUv+r/uLrW7lpi5d6uY2fQNtf0jX9/qwHUfsNJTmaoYVfKDzU2FPUbLL4l0SZ9kmC2NS2vC2EL2fo4fhCwn0/UQ="
express_bp_str = "0eJydkttugzAMht/F18lU6Glw2deYpim0HrMEJkrM1Arx7guniRXUTbuKEtnf9ydxA1lRo3XEAmkDdK7YQ/rSgKecTdGdyc0ipECCJShgU3Y74z2WWUGc69KcP4hRR9AqIL7gFdKofVWALCSEA2/sezdeNLFHJ+gC7kIOz0IVQ3pQYCtPw6aBQNk87RXc+rUdcbc3rssstAaFah6k2f6EbxZwHT2ixwpCLw03n+LOjHi1Dr3X4gx7WznRGRbym/OhcruC97YgWbzUEhz32GgNu1vB1uGXXO6qsE65718nHqPqATqzB9c0EmzrrvfOuP+X8S/CqpZV46Ebt35A09k8K/hE5wfEc7Q7JvFxFyfJNokUFCakCNWn7+q2/QJ+owLE"


# class TestThreeAssemblers(unittest.TestCase):
#     N = 10
#     def setUp(self):
#         self.factory = Factory.from_str(threeassemblers)
#         self.json = self.factory.blueprint.to_dict()
#         assemblers = [
#             e for e in self.json['blueprint']['entities'] if e['name'].startswith('assembling')
#         ]
#         left, top = bound_bp_by_json(self.json)
#         for entity in assemblers:
#             entity['position']['x'] -= left
#             entity['position']['y'] -= top
            
#         self.amat = make_assembler_matrix(assemblers, self.N, self.N)

class TestSimpleMatrix(unittest.TestCase):
    N = 10
    def setUp(self):
        self.factory = Factory.from_str(simple_bp_str)
        self.json = self.factory.blueprint.to_dict()
        assemblers = [
            e for e in self.json['blueprint']['entities'] if e['name'].startswith('assembling')
        ]
        left, top = bound_bp_by_json(self.json)
        for entity in assemblers:
            entity['position']['x'] -= left
            entity['position']['y'] -= top
            
        self.amat = make_assembler_matrix(assemblers, self.N, self.N)

    def test_make_assembler_matrix(self):
        for k in ['assembler', 'item', 'recipe']:
            self.assertIn(k, self.amat)

        self.assertEqual(self.amat['assembler'].shape, (self.N, self.N, 1))
        self.assertEqual(self.amat['item'].shape, (self.N, self.N, 3))
        self.assertEqual(self.amat['recipe'].shape, (self.N, self.N, 5))

    def test_opacity_matrix_integrity(self):
        omat = self.amat['assembler']
        row_sums = omat.sum(axis=1)
        col_sums = omat.sum(axis=0)
        self.assertEqual(len(row_sums), 10)
        self.assertEqual(len(col_sums), 10)
        self.assertEqual(omat.sum(), 9)

    def test_recipe_matrix_integerity(self):
        recipes = self.amat['recipe']
        self.assertEqual(recipes.sum(), 9)
        self.assertEqual(recipes[:,:,0].sum(), 9)
        self.assertEqual(recipes[:,:,1].sum(), 0)
        self.assertEqual(recipes[:,:,2].sum(), 0)
        self.assertEqual(recipes[:,:,3].sum(), 0)
        self.assertEqual(recipes[:,:,4].sum(), 0)

    def test_item_id_matrix(self):
        itemid = self.amat['item']
        self.assertEqual(itemid.sum(), 9)
        self.assertEqual(itemid[:,:,0].sum(), 9)
        self.assertEqual(itemid[:,:,1].sum(), 0)
        self.assertEqual(itemid[:,:,2].sum(), 0)


class TestFastAssemblerMatrix(unittest.TestCase):
    N = 10
    def setUp(self):
        self.factory = Factory.from_str(fast_bp_str)
        self.json = self.factory.blueprint.to_dict()
        assemblers = [
            e for e in self.json['blueprint']['entities'] if e['name'].startswith('assembling')
        ]
        left, top = bound_bp_by_json(self.json)
        for entity in assemblers:
            entity['position']['x'] -= left
            entity['position']['y'] -= top
            
        self.amat = make_assembler_matrix(assemblers, self.N, self.N)

    def test_make_assembler_matrix(self):
        for k in ['assembler', 'item', 'recipe']:
            self.assertIn(k, self.amat)

        self.assertEqual(self.amat['assembler'].shape, (self.N, self.N, 1))
        self.assertEqual(self.amat['item'].shape, (self.N, self.N, 3))
        self.assertEqual(self.amat['recipe'].shape, (self.N, self.N, 5))

    def test_opacity_matrix_integrity(self):
        omat = self.amat['assembler']
        row_sums = omat.sum(axis=1)
        col_sums = omat.sum(axis=0)
        self.assertEqual(len(row_sums), 10)
        self.assertEqual(len(col_sums), 10)
        self.assertEqual(omat.sum(), 9)

    def test_recipe_matrix_integerity(self):
        recipes = self.amat['recipe']
        self.assertEqual(recipes.sum(), 9)
        self.assertEqual(recipes[:,:,0].sum(), 9)
        self.assertEqual(recipes[:,:,1].sum(), 0)
        self.assertEqual(recipes[:,:,2].sum(), 0)
        self.assertEqual(recipes[:,:,3].sum(), 0)
        self.assertEqual(recipes[:,:,4].sum(), 0)

    def test_item_id_matrix(self):
        itemid = self.amat['item']
        self.assertEqual(itemid.sum(), 9)
        self.assertEqual(itemid[:,:,0].sum(), 0)
        self.assertEqual(itemid[:,:,1].sum(), 9)
        self.assertEqual(itemid[:,:,2].sum(), 0)


class TestExpressAssemblerMatrix(unittest.TestCase):
    N = 10
    def setUp(self):
        self.factory = Factory.from_str(express_bp_str)
        self.json = self.factory.blueprint.to_dict()
        assemblers = [
            e for e in self.json['blueprint']['entities'] if e['name'].startswith('assembling')
        ]
        left, top = bound_bp_by_json(self.json)
        for entity in assemblers:
            entity['position']['x'] -= left
            entity['position']['y'] -= top
            
        self.amat = make_assembler_matrix(assemblers, self.N, self.N)

    def test_make_assembler_matrix(self):
        for k in ['assembler', 'item', 'recipe']:
            self.assertIn(k, self.amat)

        self.assertEqual(self.amat['assembler'].shape, (self.N, self.N, 1))
        self.assertEqual(self.amat['item'].shape, (self.N, self.N, 3))
        self.assertEqual(self.amat['recipe'].shape, (self.N, self.N, 5))

    def test_opacity_matrix_integrity(self):
        omat = self.amat['assembler']
        row_sums = omat.sum(axis=1)
        col_sums = omat.sum(axis=0)
        self.assertEqual(len(row_sums), 10)
        self.assertEqual(len(col_sums), 10)
        self.assertEqual(omat.sum(), 9)

    def test_recipe_matrix_integerity(self):
        recipes = self.amat['recipe']
        self.assertEqual(recipes.sum(), 9)
        self.assertEqual(recipes[:,:,0].sum(), 9)
        self.assertEqual(recipes[:,:,1].sum(), 0)
        self.assertEqual(recipes[:,:,2].sum(), 0)
        self.assertEqual(recipes[:,:,3].sum(), 0)
        self.assertEqual(recipes[:,:,4].sum(), 0)

    def test_item_id_matrix(self):
        itemid = self.amat['item']
        self.assertEqual(itemid.sum(), 9)
        self.assertEqual(itemid[:,:,0].sum(), 0)
        self.assertEqual(itemid[:,:,1].sum(), 0)
        self.assertEqual(itemid[:,:,2].sum(), 9)


if __name__ == "__main__":
    unittest.main()