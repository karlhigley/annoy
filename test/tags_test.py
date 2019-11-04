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
        n_tags = 3
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0])
        i.add_item_with_tags(1, [3, 2], [1])
        i.add_item_with_tags(2, [3, 3], [2])
        i.build(10)

        self.assertEqual(i.get_n_items(), 3)

    def test_get_item_tags(self):
        f = 2
        n_tags = 3
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0])
        i.add_item_with_tags(1, [3, 2], [1])
        i.add_item_with_tags(2, [3, 3], [2])
        i.build(10)

        self.assertEqual(i.get_item_tags(0), [0])
        self.assertEqual(i.get_item_tags(1), [1])
        self.assertEqual(i.get_item_tags(2), [2])

    def test_get_nns_by_item_and_tags(self):
        f = 2
        n_tags = 3
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [0])
        i.add_item_with_tags(1, [3, 2], [1])
        i.add_item_with_tags(2, [3, 3], [2])
        i.build(10)
        
        expected_ids = [2, 1, 0]
        search_tag = 0
        num_neighbors = 3

        self.assertEqual(i.get_nns_by_item(0, num_neighbors), [2, 1, 0])
        self.assertEqual(i.get_nns_by_item(2, num_neighbors), [2, 1, 0])

        self.assertEqual(i.get_nns_by_item_and_tags(0, [1], num_neighbors), [1])
        self.assertEqual(i.get_nns_by_item_and_tags(2, [1], num_neighbors), [1])

    def test_get_nns_by_vector_and_tags(self):
        f = 2
        n_tags = 3
        i = AnnoyIndex(f, 'dot', n_tags)
        i.add_item_with_tags(0, [2, 2], [1])
        i.add_item_with_tags(1, [3, 2], [1])
        i.add_item_with_tags(2, [3, 3], [0])
        i.build(10)
        
        num_neighbors = 3
        expected_ids = [2, 1, 0]
        

        self.assertEqual(i.get_nns_by_vector([4, 4], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([1, 1], num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector([4, 2], num_neighbors), expected_ids)

        search_tag = [1]
        expected_ids = [1,0]

        self.assertEqual(i.get_nns_by_vector_and_tags([4, 4], search_tag, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector_and_tags([1, 1], search_tag, num_neighbors), expected_ids)
        self.assertEqual(i.get_nns_by_vector_and_tags([4, 2], search_tag, num_neighbors), expected_ids)
