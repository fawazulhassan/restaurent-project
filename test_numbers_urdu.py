import unittest

from app.numbers_urdu import number_to_urdu_words
from app.order import Order, add_item_to_order, build_confirmation_urdu
from app.replies import build_verified_summary_urdu


class UrduNumberTests(unittest.TestCase):
    def test_menu_prices_and_totals(self) -> None:
        expected = {
            0: "صفر",
            1: "ایک",
            50: "پچاس",
            80: "اسی",
            200: "دو سو",
            250: "دو سو پچاس",
            350: "تین سو پچاس",
            800: "آٹھ سو",
            1200: "ایک ہزار دو سو",
            1400: "ایک ہزار چار سو",
            2200: "دو ہزار دو سو",
            1480: "ایک ہزار چار سو اسی",
            2800: "دو ہزار آٹھ سو",
        }
        for value, words in expected.items():
            with self.subTest(value=value):
                self.assertEqual(number_to_urdu_words(value), words)

    def test_large_values(self) -> None:
        self.assertEqual(number_to_urdu_words(100_000), "ایک لاکھ")
        self.assertEqual(
            number_to_urdu_words(12_345_678),
            "ایک کروڑ تئیس لاکھ پینتالیس ہزار چھ سو اٹھہتر",
        )

    def test_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            number_to_urdu_words(-1)
        with self.assertRaises(TypeError):
            number_to_urdu_words(1.5)  # type: ignore[arg-type]

    def test_customer_facing_summaries_do_not_speak_digits(self) -> None:
        order = Order(delivery_address="ماڈل ٹاؤن قصور")
        add_item_to_order(order, "chicken-karahi", "full", 1)
        add_item_to_order(order, "garlic-naan", None, 1)

        summary = build_verified_summary_urdu(
            order, ask_for_confirmation=True
        )
        confirmation = build_confirmation_urdu(order)
        for reply in (summary, confirmation):
            self.assertIn("ایک ہزار چار سو اسی روپے", reply)
            self.assertNotRegex(reply, r"[0-9]")
            self.assertNotIn("Rs", reply)


if __name__ == "__main__":
    unittest.main()
