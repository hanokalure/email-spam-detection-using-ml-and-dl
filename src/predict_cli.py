import argparse
import sys
from simple_svm_classifier import SimpleSVMClassifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a trained SVM spam model and predict messages."
    )
    parser.add_argument(
        "--model", "-m", required=True, help="Path to the saved model .pkl file"
    )
    parser.add_argument(
        "--text", "-t", help="Single message to classify (if not provided, interactive mode starts)"
    )
    parser.add_argument(
        "--batch-file", "-f", help="Path to a text file with one message per line to classify in batch"
    )
    args = parser.parse_args()

    # Load model
    clf = SimpleSVMClassifier()
    clf.load_model(args.model)

    def classify(msg: str):
        pred, conf = clf.predict(msg)
        label = "SPAM" if pred == 1 else "HAM"
        print(f"{label}\tconf={abs(conf):.3f}\tmsg={msg}")

    # Single text mode
    if args.text:
        classify(args.text)
        return 0

    # Batch file mode
    if args.batch_file:
        try:
            with open(args.batch_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    msg = line.strip()
                    if msg:
                        classify(msg)
        except FileNotFoundError:
            print(f"Batch file not found: {args.batch_file}", file=sys.stderr)
            return 1
        return 0

    # Interactive mode
    print("Interactive mode. Type a message (or 'quit' to exit):")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if msg:
            classify(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
