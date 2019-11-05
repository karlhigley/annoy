# Copyright (c) 2019 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy
import random
from common import TestCase
from annoy import AnnoyIndex

class TagsTest(TestCase):
    def test_get_n_tags(self):
        f = 2
        n_tags = 2
        i = AnnoyIndex(f, 'dot', n_tags)

        self.assertEqual(i.get_n_tags(), n_tags)

    def test_add_item_with_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0,1,2])
        i.add_item_with_tags(1, [3, 2], [3,4,5])
        i.add_item_with_tags(2, [3, 3], [6,7,8])
        i.build(10)

        self.assertEqual(i.get_n_items(), 3)

    def test_get_item_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0,1,2])
        i.add_item_with_tags(1, [3, 2], [3,4,5])
        i.add_item_with_tags(2, [3, 3], [6,7,8])
        i.build(10)

        self.assertEqual(i.get_item_tags(0), [0,1,2])
        self.assertEqual(i.get_item_tags(1), [3,4,5])
        self.assertEqual(i.get_item_tags(2), [6,7,8])

    def test_get_nns_by_item_and_any_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)

        i.add_item_with_tags(0, [2, 2], [0,1,2])
        i.add_item_with_tags(1, [3, 2], [2,3,4])
        i.add_item_with_tags(2, [3, 3], [4,5,6])
        i.build(10)
        
        expected_ids = [2, 1, 0]
        num_neighbors = 3

        self.assertEqual(i.get_nns_by_item(0, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_item(1, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_item(2, num_neighbors), expected_ids)

        # Single tag match any
        self.assertEqual(i.get_nns_by_item_and_tags(0, [3], num_neighbors), [1])
        self.assertEqual(i.get_nns_by_item_and_tags(1, [5], num_neighbors), [2])     
        self.assertEqual(i.get_nns_by_item_and_tags(2, [1], num_neighbors), [0])

        self.assertEqual(i.get_nns_by_item_and_tags(2, [2], num_neighbors), [1,0])

        # Multi-tag match any
        self.assertEqual(i.get_nns_by_item_and_tags(0, [3,9], num_neighbors), [1])
        self.assertEqual(i.get_nns_by_item_and_tags(1, [5,9], num_neighbors), [2])
        self.assertEqual(i.get_nns_by_item_and_tags(2, [1,9], num_neighbors), [0])

        self.assertEqual(i.get_nns_by_item_and_tags(2, [2,4], num_neighbors), [2,1,0])

    def test_get_nns_by_item_and_all_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)

        i.add_item_with_tags(0, [2, 2], [0,1,2])
        i.add_item_with_tags(1, [3, 2], [2,3,4])
        i.add_item_with_tags(2, [3, 3], [4,5,6])
        i.build(10)
        
        expected_ids = [2, 1, 0]
        num_neighbors = 3

        self.assertEqual(i.get_nns_by_item(0, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_item(1, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_item(2, num_neighbors), expected_ids)

        # Single tag match all
        self.assertEqual(i.get_nns_by_item_and_tags(0, [3], num_neighbors, match_all=True), [1])
        self.assertEqual(i.get_nns_by_item_and_tags(1, [5], num_neighbors, match_all=True), [2])     
        self.assertEqual(i.get_nns_by_item_and_tags(2, [1], num_neighbors, match_all=True), [0])

        self.assertEqual(i.get_nns_by_item_and_tags(2, [2], num_neighbors, match_all=True), [1,0])

        # Multi-tag match all
        self.assertEqual(i.get_nns_by_item_and_tags(0, [3,9], num_neighbors, match_all=True), [])
        self.assertEqual(i.get_nns_by_item_and_tags(0, [3,4], num_neighbors, match_all=True), [1])

        self.assertEqual(i.get_nns_by_item_and_tags(1, [5,9], num_neighbors, match_all=True), [])
        self.assertEqual(i.get_nns_by_item_and_tags(1, [5,6], num_neighbors, match_all=True), [2])

        self.assertEqual(i.get_nns_by_item_and_tags(2, [1,9], num_neighbors, match_all=True), [])
        self.assertEqual(i.get_nns_by_item_and_tags(2, [1,0], num_neighbors, match_all=True), [0])

        self.assertEqual(i.get_nns_by_item_and_tags(2, [2,4], num_neighbors, match_all=True), [1])    

    def test_get_nns_by_vector_and_any_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0,1,2])
        i.add_item_with_tags(1, [3, 2], [2,3,4])
        i.add_item_with_tags(2, [3, 3], [4,5,6])
        i.build(10)
        
        num_neighbors = 3
        expected_ids = [2, 1, 0]

        self.assertEqual(i.get_nns_by_vector([4, 4], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([1, 1], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([4, 2], num_neighbors), expected_ids)

        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [2], num_neighbors), [1,0])
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [2,9], num_neighbors), [1,0])
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [0,3], num_neighbors), [1,0])

    def test_get_nns_by_vector_and_all_tags(self):
        f = 2
        n_tags = 10
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0,1,2,9])
        i.add_item_with_tags(1, [3, 2], [2,3,4,9])
        i.add_item_with_tags(2, [3, 3], [4,5,6,9])
        i.build(10)
        
        num_neighbors = 3
        expected_ids = [2, 1, 0]

        self.assertEqual(i.get_nns_by_vector([4, 4], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([1, 1], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([4, 2], num_neighbors), expected_ids) 

        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [2], num_neighbors, match_all=True), [1,0])
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [2,9], num_neighbors, match_all=True), [1,0])
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [0,2], num_neighbors, match_all=True), [0])
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], [0,3], num_neighbors, match_all=True), [])