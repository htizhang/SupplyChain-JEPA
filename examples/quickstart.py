from supplychain_jepa import load_pretrained


def main() -> None:
    bundle = load_pretrained(device="cpu")
    print(bundle.config["model_type"])
    print(sum(parameter.numel() for parameter in bundle.model.parameters()))


if __name__ == "__main__":
    main()
