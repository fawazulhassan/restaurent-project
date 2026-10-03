import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from app.dialog import (
    _is_menu_inquiry,
    build_spoken_menu_urdu,
    build_system_prompt,
    chat_turn,
    normalize_menu_aliases,
)
from app.order import Order, add_item_to_order


class SpokenMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.menu = json.loads(
            Path("data/menu.json").read_text(encoding="utf-8")
        )

    def test_menu_json_has_unique_item_ids(self) -> None:
        ids = [
            item["id"]
            for category in self.menu["categories"]
            for item in category["items"]
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_spoken_menu_uses_clean_urdu_labels(self) -> None:
        reply = build_spoken_menu_urdu(self.menu)

        self.assertIn("لہسن والا نان", reply)
        self.assertIn("آدھی آٹھ سو روپے", reply)
        self.assertIn("ایک پلیٹ تین سو پچاس روپے", reply)
        self.assertIn("ڈیڑھ لیٹر کوک", reply)
        self.assertNotIn("Chicken", reply)
        self.assertNotIn("Single", reply)
        self.assertNotIn("- ", reply)
        self.assertIsNone(re.search(r"[\u0400-\u04ff\u1100-\u11ff]", reply))

    def test_common_transcripts_are_detected_as_menu_questions(self) -> None:
        questions = (
            "آپ کے مینی میں کیا چیزیں ہیں",
            "منیو بتا دیں",
            "What do you have on the menu?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(_is_menu_inquiry(question))

    def test_menu_question_does_not_call_llm(self) -> None:
        order = Order()
        messages = [{"role": "system", "content": build_system_prompt()}]

        with patch("app.dialog._call_llm") as call_llm:
            reply, returned_order, returned_messages, complete = chat_turn(
                "آپ کے مینی میں کیا چیزیں ہیں",
                order,
                messages,
            )

        call_llm.assert_not_called()
        self.assertIn("چکن کڑاہی", reply)
        self.assertIs(returned_order, order)
        self.assertEqual(returned_messages[-1]["content"], reply)
        self.assertFalse(complete)

    def test_misheard_garlic_naan_is_handled_without_llm(self) -> None:
        self.assertEqual(
            normalize_menu_aliases("مجھے ایک کالکنان چاہیے"),
            "مجھے ایک لہسن والا نان چاہیے",
        )

        order = Order()
        messages = [{"role": "system", "content": build_system_prompt()}]
        with patch("app.dialog.resolve_ambiguous_intent") as fallback:
            reply, returned_order, _, complete = chat_turn(
                "مجھے ایک کالکنان چاہیے", order, messages
            )

        fallback.assert_not_called()
        self.assertEqual(returned_order.items[0].id, "garlic-naan")
        self.assertTrue(reply)
        self.assertFalse(complete)

    def test_order_confirmation_data_uses_urdu_size_label(self) -> None:
        order = Order()
        add_item_to_order(order, "chicken-karahi", "half", 1)

        self.assertEqual(order.items[0].name_urdu, "چکن کڑاہی")
        self.assertEqual(order.items[0].size_label, "آدھی")


if __name__ == "__main__":
    unittest.main()
