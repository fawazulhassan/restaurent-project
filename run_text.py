from app.cli_utils import configure_stdout, print_order_summary
from app.dialog import LLMRateLimitError, build_system_prompt, chat_turn
from app.order import Order, build_confirmation_english, save_order


def main() -> None:
    configure_stdout()

    # print("السلام علیکم! Kasur Kitchen میں خوش آمدید۔")
    # print("اپنا آرڈر لکھیں (Roman Urdu یا English)۔ ختم کرنے کے لیے 'quit' لکھیں۔\n")
    print("Assalam o alaikum! Welcome to Kasur Kitchen.")
    print("Type your order (Roman Urdu or English). Type 'quit' to exit.\n")

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        try:
            reply, order, messages, is_complete = chat_turn(
                user_input, order, messages
            )
        except LLMRateLimitError as e:
            print(f"\nRate limit: {e}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAI: {reply}\n")
        print_order_summary(order)

        if is_complete:
            path = save_order(order)
            print(f"Order saved: {path}")
            print(f"\n{build_confirmation_english(order)}\n")
            break


if __name__ == "__main__":
    main()
