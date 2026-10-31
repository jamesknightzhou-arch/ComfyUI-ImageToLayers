import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layout import compose_layers, fit_within, layer_box, sort_bottom_to_top  # noqa: E402


def png(size, color):
    out = io.BytesIO()
    Image.new("RGBA", size, color).save(out, format="PNG")
    return out.getvalue()


class LayerBox(unittest.TestCase):
    def test_background_has_no_box(self):
        self.assertIsNone(layer_box({"id": "a", "bbox": None}))
        self.assertIsNone(layer_box({"id": "a"}))

    def test_xywh_and_absolute(self):
        self.assertEqual(layer_box({"bbox": {"x": 10, "y": 20, "width": 30, "height": 40}}), (10, 20, 30, 40))
        self.assertEqual(layer_box({"bbox": {"absolute": [10, 20, 40, 60]}}), (10, 20, 30, 40))

    def test_degenerate_boxes_are_ignored(self):
        self.assertIsNone(layer_box({"bbox": {"x": 1, "y": 1, "width": 0, "height": 5}}))


class Sorting(unittest.TestCase):
    def test_by_z_index(self):
        layers = [{"id": "c", "bbox": {"zIndex": 2}}, {"id": "a", "bbox": {"zIndex": 0}}, {"id": "b", "bbox": {"zIndex": 1}}]
        self.assertEqual([l["id"] for l in sort_bottom_to_top(layers)], ["a", "b", "c"])

    def test_without_z_the_server_lists_top_first(self):
        layers = [{"id": "top"}, {"id": "mid"}, {"id": "bg"}]
        self.assertEqual([l["id"] for l in sort_bottom_to_top(layers)], ["bg", "mid", "top"])


class Fit(unittest.TestCase):
    def test_keeps_proportions_and_centres(self):
        self.assertEqual(fit_within((0, 0, 100, 100), (50, 25)), (0, 25, 100, 50))
        self.assertEqual(fit_within((10, 10, 100, 100), (25, 50)), (35, 10, 50, 100))


class Compose(unittest.TestCase):
    def test_places_layer_on_background_canvas(self):
        layers = [{"name": "Background", "bbox": None}, {"name": "Dot", "bbox": {"x": 2, "y": 3, "width": 4, "height": 4}}]
        arrays, names = compose_layers(layers, [png((10, 8), (0, 0, 255, 255)), png((4, 4), (255, 0, 0, 255))])
        self.assertEqual(names, ["Background", "Dot"])
        self.assertEqual(arrays[0].shape, (8, 10, 4))
        self.assertEqual(arrays[1].shape, (8, 10, 4))
        self.assertEqual(arrays[1][3, 2, 3], 1.0)  # inside the dot: opaque
        self.assertEqual(arrays[1][0, 0, 3], 0.0)  # outside: transparent
        self.assertEqual(arrays[0][0, 0, 2], 1.0)  # background is blue

    def test_canvas_from_boxes_without_a_plate(self):
        layers = [{"name": None, "bbox": {"x": 0, "y": 0, "width": 6, "height": 4}}]
        arrays, names = compose_layers(layers, [png((6, 4), (0, 255, 0, 255))])
        self.assertEqual(arrays[0].shape, (4, 6, 4))
        self.assertEqual(names, ["Layer 1"])

    def test_a_layer_with_an_unusable_box_is_the_background_plate(self):
        arrays, _ = compose_layers([{"bbox": {"x": 0, "y": 0, "width": 0, "height": 0}, "name": "x"}], [png((3, 2), (0, 0, 0, 255))])
        self.assertEqual(arrays[0].shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
