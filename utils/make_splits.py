import argparse
import os


from label_formats import get_ids as get_ids_from_label_file


def write_split_file(ids, path):
    with open(path, "w") as f:
        for record_id in ids:
            f.write(f"{record_id}\n")
    print(f"  wrote {len(ids)} ids -> {path}")


def train_split_filename(train_file_paths):
    stems = [os.path.splitext(os.path.basename(p))[0] for p in train_file_paths]
    return f"train_split_{'_'.join(stems)}.txt"


def test_split_filename(test_file_path):
    """test_split_{stem}.txt, e.g. 'DNA-129_Test.txt' -> 'test_split_DNA-129_Test.txt'"""
    stem = os.path.splitext(os.path.basename(test_file_path))[0]
    return f"test_split_{stem}.txt"


def main():
    parser = argparse.ArgumentParser(description="Create train/test split files for a task")
    parser.add_argument("--train_label_files", nargs="+", required=True,
                         help="One or more training label .txt/.fa files")
    parser.add_argument("--test_label_files", nargs="+", required=True,
                         help="One or more test label .txt/.fa files. Each gets "
                              "its own separate test split file.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write split files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Train IDs
    # --------------------------------------------------
    train_ids = []
    for path in args.train_label_files:
        ids = get_ids_from_label_file(path)
        print(f"{os.path.basename(path)}: {len(ids)} ids")
        train_ids.extend(ids)
    train_ids = list(dict.fromkeys(train_ids))  # de-dup, preserve order

    print(f"\nTotal train ids: {len(train_ids)}")

    # --------------------------------------------------
    # Test IDs -- one split file PER test label file
    # --------------------------------------------------
    for test_path in args.test_label_files:
        ids = get_ids_from_label_file(test_path)
        ids = list(dict.fromkeys(ids))  # de-dup within this file
        print(f"{os.path.basename(test_path)}: {len(ids)} ids")

        out_name = test_split_filename(test_path)
        write_split_file(ids, os.path.join(args.output_dir, out_name))

    # --------------------------------------------------
    # Write train_split.txt
    # --------------------------------------------------
    train_name = train_split_filename(args.train_label_files)
    write_split_file(train_ids, os.path.join(args.output_dir, train_name))


if __name__ == "__main__":
    main()