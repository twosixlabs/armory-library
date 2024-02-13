# header
# I'd appreciate suggestions on what I should include in/ how I should format my header


def main():
    string_1, integer_1, integer_2 = (
        input("Enter string_1: "),
        int(input("Enter integer_1: ")),
        int(input("Enter integer_2: ")),
    )
    print(f"\nLorem ipsum {string_1} {integer_1}. Dolor sit {integer_2 + 5} amet.")


if __name__ == "__main__":
    main()
