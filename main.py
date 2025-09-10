import argparse

def main():


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Autora workflow with specified parameters.")
    parser.add_argument('-c', '--config_json', type=int, default=4, help='Maximum depth of the generated equation.')

    args = parser.parse_args()

    main(args)