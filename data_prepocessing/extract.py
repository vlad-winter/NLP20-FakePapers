from LatexTokenizer import file_to_tree
from LatexCleaning import find_wrapper_node, to_text
import argparse
import os


def process_papers(path):
    data_file_names = os.listdir(path)
    for f_name in data_file_names:
        f_path = os.path.join(path, f_name)

        if not os.path.isfile(f_path):
            print(f"not a file: {f_name}")
            break
        try:
            t = file_to_tree(f_path)

            d = find_wrapper_node(t, "document")

            abstract = find_wrapper_node(d, "abstract")
            if abstract is None:
                print(f"Error with abstract extraction: {f_name}")
                continue
            else:
                tx_abs = to_text(abstract)

            intro = find_wrapper_node(d, "Introduction")

            if intro is None:
                print(f"Error with Introduction extraction: {f_name}")
                continue
            else:
                tx_intro = to_text(intro)

            paper_name = os.path.splitext(f_name)[0]
            paper_dir = os.path.join("./clean_dataset", paper_name)
            try:
                os.mkdir(paper_dir)
            except FileExistsError:
                pass

            abs_file = open(os.path.join(paper_dir, "abs.txt"), "w")
            abs_file.write(tx_abs)
            abs_file.close()

            intro_file = open(os.path.join(paper_dir, "intro.txt"), "w")
            intro_file.write(tx_intro)
            intro_file.close()

            print(f"finished {f_name}")

        except Exception as e:
            print(e, f_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", action="store", help="Path to papers dir")

    args = parser.parse_args()

    process_papers(args.path)
