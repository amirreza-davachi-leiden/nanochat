"""Talk to a saved Nanochat checkpoint from the terminal.

This is the only Task 4 script for interactive chat. It does not train,
save a transcript, or modify model weights.
"""

import argparse
import gc
import json
import os
import shlex

import common


def profile_spec(current_spec, profile):
    """Resolve one named model while retaining device and cache settings."""
    required = ("checkpoint_dir", "tokenizer_dir", "step", "stage")
    missing = [field for field in required if field not in profile]
    if missing:
        raise ValueError(f"Model profile is missing: {', '.join(missing)}")
    if not isinstance(profile["step"], int) or profile["step"] < 0:
        raise ValueError("Model profile step must be a nonnegative integer.")
    if profile["stage"] not in ("base", "mid", "sft"):
        raise ValueError("Model profile stage must be base, mid, or sft.")
    selected = dict(current_spec)
    selected.update({
        "checkpoint_dir": common.path_from_repo(profile["checkpoint_dir"]),
        "tokenizer_dir": common.path_from_repo(profile["tokenizer_dir"]),
        "step": profile["step"],
        "stage": profile["stage"],
    })
    return selected


def same_checkpoint(left, right):
    return all(left[field] == right[field] for field in (
        "checkpoint_dir", "tokenizer_dir", "step", "stage"
    ))


def show_models(profiles, current_name, current_spec):
    if not profiles:
        print("No named models are configured in task4/config.json.")
        return
    print("Available model profiles (* = loaded):")
    for number, (name, profile) in enumerate(profiles.items(), start=1):
        try:
            choice = profile_spec(current_spec, profile)
            checkpoint = choice["checkpoint_dir"] / f"model_{choice['step']:06d}.pt"
            tokenizer = choice["tokenizer_dir"] / "tokenizer.pkl"
            availability = "ready" if checkpoint.is_file() and tokenizer.is_file() else "files missing"
            marker = "*" if name == current_name else " "
            print(f" {marker} {number}. {name:5} ({choice['stage']}, step {choice['step']}; {availability})")
        except (TypeError, ValueError, KeyError) as error:
            print(f"   {number}. {name:5} (invalid profile: {error})")
    print("Switch with :model NAME or :model NUMBER; switching clears chat history.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_options(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Load checkpoint/tokenizer, generate one token, print identity, and exit.",
    )
    args = parser.parse_args()

    config, spec, generation, _ = common.configuration(args)
    profiles = config.get("chat_models", {})
    if not isinstance(profiles, dict):
        raise ValueError("chat_models in the config must be an object of named profiles.")
    identity = common.verify_files(spec)
    model, tokenizer, engine = common.load_nanochat(spec, identity)
    if args.check_only:
        ids = common.chat_prefix(tokenizer, "Hi.")
        common.context_check(model, ids, 1)
        result = common.generate_reply(
            engine, tokenizer, ids,
            temperature=0.0, top_k=0, max_new_tokens=1, seed=42,
        )
        print(json.dumps({
            "checkpoint": identity,
            "one_token_load_test": {
                "generated_id": result["generated_ids"][0],
                "generated_text": result["response"],
            },
        }, indent=2, ensure_ascii=False))
        return

    max_new, top_k = common.generation_settings(args, generation)
    if args.temperature < 0 or args.seed < 0:
        raise ValueError("Temperature and seed must be nonnegative.")
    history = [tokenizer.get_bos_token_id()]
    turn = 0
    current_name = next(
        (name for name, profile in profiles.items()
         if same_checkpoint(spec, profile_spec(spec, profile))),
        "custom",
    )
    print(f"Loaded {current_name} ({identity['stage']}, step {identity['step']}) on {identity['device']}.")
    print("Type :models to list checkpoints, :model base|mid|sft to switch, or :help.")
    while True:
        try:
            user_text = input(f"\nYou [{current_name}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if user_text.lower() in ("exit", "quit"):
            return
        if user_text.lower() == "clear":
            history = [tokenizer.get_bos_token_id()]
            turn = 0
            print("Conversation cleared.")
            continue
        if user_text.startswith(":"):
            try:
                command = shlex.split(user_text[1:])
            except ValueError as error:
                print(f"Invalid command: {error}")
                continue
            if not command:
                continue
            action = command[0].lower()
            if action in ("exit", "quit") and len(command) == 1:
                return
            if action == "help" and len(command) == 1:
                print(":models  list checkpoints; :model NAME or NUMBER  switch model")
                print(":current  show loaded checkpoint; :clear  reset chat; :exit  quit")
                continue
            if action in ("models", "model") and len(command) == 1:
                show_models(profiles, current_name, spec)
                continue
            if action == "current" and len(command) == 1:
                print(f"{current_name}: {identity['model_file']}")
                print(f"Tokenizer: {identity['tokenizer_dir']}")
                continue
            if action == "clear" and len(command) == 1:
                history = [tokenizer.get_bos_token_id()]
                turn = 0
                print("Conversation cleared.")
                continue
            if action == "model" and len(command) == 2:
                requested = command[1].lower()
                if requested.isdigit():
                    index = int(requested) - 1
                    if 0 <= index < len(profiles):
                        requested = list(profiles)[index]
                if requested not in profiles:
                    print(f"Unknown model {command[1]!r}. Type :models to see choices.")
                    continue
                previous_environment = {
                    key: os.environ.get(key)
                    for key in ("NANOCHAT_BASE_DIR", "NANOCHAT_DTYPE")
                }
                try:
                    choice = profile_spec(spec, profiles[requested])
                    if same_checkpoint(spec, choice):
                        print(f"{requested} is already loaded. Use :clear for a fresh chat.")
                        continue
                    candidate_identity = common.verify_files(choice)
                    candidate_model, candidate_tokenizer, candidate_engine = common.load_nanochat(
                        choice, candidate_identity
                    )
                except Exception as error:
                    # Keep the previous model and history if the selected files fail.
                    for key, old_value in previous_environment.items():
                        if old_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = old_value
                    print(f"Could not switch models: {error}")
                    continue
                old_model, old_tokenizer, old_engine = model, tokenizer, engine
                model, tokenizer, engine = candidate_model, candidate_tokenizer, candidate_engine
                spec, identity, current_name = choice, candidate_identity, requested
                history = [tokenizer.get_bos_token_id()]
                turn = 0
                del old_model, old_tokenizer, old_engine
                del candidate_model, candidate_tokenizer, candidate_engine
                gc.collect()
                print(f"Switched to {current_name} (step {identity['step']}). Conversation cleared.")
                if current_name == "base":
                    print("Note: the base model was not trained to follow chat instructions.")
                continue
            print("Unknown command. Type :help for the menu.")
            continue
        if not user_text:
            continue

        # Preserve the prior turns as IDs; each new answer gets a fresh KV cache.
        ids = common.chat_prefix(tokenizer, user_text, history)
        try:
            common.context_check(model, ids, max_new)
        except ValueError as error:
            print(error)
            continue
        answer = common.generate_reply(
            engine, tokenizer, ids, temperature=args.temperature,
            top_k=top_k, max_new_tokens=max_new, seed=args.seed + turn,
        )
        print(f"Assistant: {answer['response']}")
        history = ids + answer["generated_ids"]
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        if not history or history[-1] != assistant_end:
            history.append(assistant_end)
        turn += 1


if __name__ == "__main__":
    main()
