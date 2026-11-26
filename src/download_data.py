import os
import sys
import shutil
import zipfile
import tarfile
import urllib.request
import argparse


DATASETS = {
    "cifar10": {
        "url": "https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.zip",
        "archive_name": "cifar10.zip",
        "type": "zip",
        "extract_folder": "CIFAR-10-images-master",
        "num_classes": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "description": "CIFAR-10 dataset (60,000 images across 10 categories, formatted as PNGs)."
    },
    "cats_and_dogs_mini": {
        "url": "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        "archive_name": "cats_and_dogs_filtered.zip",
        "type": "zip",
        "extract_folder": "cats_and_dogs_filtered",
        "num_classes": 2,
        "classes": ["cats", "dogs"],
        "description": "Google's Cats vs Dogs filtered subset (3,000 images, ~68MB). Fast download & ideal for binary testing."
    }
}


def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()


def download_and_extract(dataset_key: str, dest_root: str):
    info = DATASETS[dataset_key]
    os.makedirs(dest_root, exist_ok=True)
    archive_path = os.path.join(dest_root, info["archive_name"])
    final_dataset_dir = os.path.join(dest_root, dataset_key)

    if os.path.exists(final_dataset_dir):
        print(f"[Dataset] Folder '{final_dataset_dir}' already exists.")
        return final_dataset_dir, info["num_classes"]

    # 1. Download
    print(f"\n--- Downloading '{dataset_key}' ---")
    print(f"URL: {info['url']}")
    urllib.request.urlretrieve(info["url"], archive_path, reporthook=download_progress_hook)
    print("\nDownload complete. Extracting files...")

    # 2. Extract
    extract_temp = os.path.join(dest_root, "temp_extract")
    os.makedirs(extract_temp, exist_ok=True)

    if info["type"] == "zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)

    # 3. Structure into ImageFolder format (train / val)
    raw_folder = os.path.join(extract_temp, info["extract_folder"])
    os.makedirs(final_dataset_dir, exist_ok=True)

    if dataset_key == "cifar10":
        # CIFAR-10 has 'train' and 'test' -> rename 'test' to 'val'
        shutil.move(os.path.join(raw_folder, "train"), os.path.join(final_dataset_dir, "train"))
        shutil.move(os.path.join(raw_folder, "test"), os.path.join(final_dataset_dir, "val"))
    elif dataset_key == "cats_and_dogs_mini":
        # Cats and dogs filtered has 'train' and 'validation' -> rename 'validation' to 'val'
        shutil.move(os.path.join(raw_folder, "train"), os.path.join(final_dataset_dir, "train"))
        shutil.move(os.path.join(raw_folder, "validation"), os.path.join(final_dataset_dir, "val"))

    # 4. Cleanup
    shutil.rmtree(extract_temp, ignore_errors=True)
    if os.path.exists(archive_path):
        os.remove(archive_path)

    print(f"\n[Success] Dataset prepared at: {final_dataset_dir}")
    print(f"Structure:")
    print(f"  {final_dataset_dir}/train ({len(os.listdir(os.path.join(final_dataset_dir, 'train')))} classes)")
    print(f"  {final_dataset_dir}/val ({len(os.listdir(os.path.join(final_dataset_dir, 'val')))} classes)")
    print(f"\nReady to train command:")
    print(f"python src/train.py --data-dir \"{final_dataset_dir}\" --num-classes {info['num_classes']}\n")

    return final_dataset_dir, info["num_classes"]


def main():
    parser = argparse.ArgumentParser(description="Download and structure datasets for custom_cv_model")
    parser.add_argument("--dataset", type=str, default="cats_and_dogs_mini", choices=list(DATASETS.keys()),
                        help="Choose preset dataset: 'cats_and_dogs_mini' (68MB, 2 classes) or 'cifar10' (160MB, 10 classes)")
    parser.add_argument("--dest", type=str, default="./data", help="Destination folder for datasets")
    args = parser.parse_args()

    download_and_extract(args.dataset, args.dest)


if __name__ == "__main__":
    main()
