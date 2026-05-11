from bootstrap import initialize
from core.registry import list_operations


def main():
    initialize()

    operations = list_operations()

    print(f"Loaded {len(operations)} operations:\n")

    for op in operations:
        print(f"{op.id} -> {op.label}")


if __name__ == "__main__":
    main()