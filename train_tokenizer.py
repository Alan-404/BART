import fire
from typing import List

def main(paths: List[str]):
    print(type(paths[0]))

if __name__ == '__main__':
    fire.Fire(main)